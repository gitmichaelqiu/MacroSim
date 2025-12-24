import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import random
from collections import deque
from typing import Dict, List, Tuple

# ==============================================================================
# CONFIGURATION & CONSTANTS
# ==============================================================================
CONSTANTS = {
    'MPC': 0.75,          # Marginal Propensity to Consume
    'ALPHA': 100,         # IS Curve Sensitivity
    'BETA': 0.5,          # Phillips Curve Slope
    'GAMMA': 0.4,         # Okun's Law Coefficient
    'NATURAL_UNEMPLOYMENT': 5.0, 
    'INFLATION_TARGET': 2.0,     
    'POTENTIAL_GROWTH': 2.0,     
    'DEPRECIATION': 0.05,        
}

# ==============================================================================
# MODULE 1: ECONOMIC ENGINE (MODEL)
# ==============================================================================
class EconomyModel:
    def __init__(self, scenario: str = "Standard"):
        self.turn = 0
        self.month = 0
        self.history = []
        self.active_event = None
        self._initialize_state(scenario)
        
        # Policy Lag Mechanism
        self.interest_rate_queue = deque([4.0, 4.0], maxlen=2)

    def _initialize_state(self, scenario: str):
        """Initialize the economy based on the selected scenario."""
        self.state = {
            'GDP_Nominal': 20000,
            'GDP_Real': 20000,
            'Potential_GDP': 20000,
            'Output_Gap': 0.0,
            'Price_Level': 100.0,
            'Inflation': 2.0,
            'Unemployment': 5.0,
            'Interest_Rate': 4.0,
            'Real_Interest_Rate': 2.0,
            'Gov_Spending': 4000,
            'Tax_Revenue': 4000,
            'Deficit': 0,
            'Debt': 20000,
            'Debt_to_GDP': 100.0,
            'Supply_Shock': 0.0,
            'Demand_Shock': 0.0,
        }

        if scenario == "Stagflation (1970s)":
            self.state['Inflation'] = 9.0
            self.state['Unemployment'] = 7.5
            self.state['Supply_Shock'] = 3.0
            
        elif scenario == "Financial Crisis (2008)":
            self.state['Inflation'] = 1.0
            self.state['Unemployment'] = 8.0
            self.state['Demand_Shock'] = -1500
            self.state['Interest_Rate'] = 0.5

        # Initial history point
        self._record_monthly_data(self.state, self.state, 0) 

    def step(self, user_inputs: Dict):
        """
        Advances the economy by one Quarter (3 Months).
        """
        self.turn += 1
        prev_state = self.state.copy()
        
        # 1. Handle Random Events (Shocks for THIS turn)
        # -----------------------------------------------
        self._trigger_random_event()
        
        # 2. Unpack Inputs
        tax_rate = user_inputs['tax_rate']
        gov_spending = user_inputs['gov_spending']
        nominal_rate = user_inputs['interest_rate']
        
        supply_investment = 100 if user_inputs['supply_reform'] else 0
        gov_spending += supply_investment

        # 3. Monetary Policy Transmission
        self.interest_rate_queue.append(nominal_rate)
        effective_nominal_rate = self.interest_rate_queue[0]
        
        # Fisher Equation
        expected_inflation = self.state['Inflation']
        real_interest_rate = effective_nominal_rate - expected_inflation

        # 4. IS Curve (Demand)
        multiplier = 1 / (1 - CONSTANTS['MPC'] * (1 - tax_rate))
        
        autonomous_consumption = 2500 
        base_investment = 1700
        investment_sensitivity = CONSTANTS['ALPHA'] * real_interest_rate
        
        # Add random noise + Event Shocks
        noise_val = np.random.normal(0, 30)
        total_demand_shock = self.state.get('Demand_Shock', 0) + noise_val
        
        base_demand = autonomous_consumption + base_investment - investment_sensitivity + gov_spending + total_demand_shock
        new_real_gdp = base_demand * multiplier

        # 5. Phillips Curve (Supply)
        potential_gdp = self.state['Potential_GDP']
        output_gap_percent = ((new_real_gdp - potential_gdp) / potential_gdp) * 100
        
        total_supply_shock = self.state.get('Supply_Shock', 0) + np.random.normal(0, 0.1)
        
        new_inflation = expected_inflation + (CONSTANTS['BETA'] * output_gap_percent) + total_supply_shock
        
        # 6. Okun's Law
        new_unemployment = CONSTANTS['NATURAL_UNEMPLOYMENT'] - (CONSTANTS['GAMMA'] * output_gap_percent)
        new_unemployment = max(0.5, new_unemployment)

        # 7. Long Run Updates
        growth_rate = CONSTANTS['POTENTIAL_GROWTH'] / 4
        if user_inputs['supply_reform']:
            growth_rate += 0.2 
            
        self.state['Potential_GDP'] *= (1 + growth_rate/100)
        
        # Debt & Deficits
        tax_revenue = new_real_gdp * tax_rate
        deficit = gov_spending - tax_revenue
        self.state['Debt'] += deficit
        self.state['Price_Level'] *= (1 + new_inflation/100)
        
        new_nominal_gdp = new_real_gdp * (self.state['Price_Level'] / 100)
        debt_to_gdp = (self.state['Debt'] / new_nominal_gdp) * 100

        # Update State
        self.state.update({
            'GDP_Real': new_real_gdp,
            'GDP_Nominal': new_nominal_gdp,
            'Output_Gap': output_gap_percent,
            'Inflation': new_inflation,
            'Unemployment': new_unemployment,
            'Interest_Rate': nominal_rate,
            'Real_Interest_Rate': real_interest_rate,
            'Gov_Spending': gov_spending,
            'Tax_Revenue': tax_revenue,
            'Deficit': deficit,
            'Debt_to_GDP': debt_to_gdp
        })
        
        # 8. Generate Monthly Interpolated History
        self._record_monthly_data(prev_state, self.state, 3)

    def _trigger_random_event(self):
        """Randomly injects shocks into the economy."""
        # Reset previous shocks slowly (mean reversion)
        self.state['Demand_Shock'] *= 0.8
        self.state['Supply_Shock'] *= 0.8
        self.active_event = None

        if random.random() < 0.25: # 25% chance per quarter
            events = [
                {"title": "Tech Breakthrough", "msg": "Productivity soars! Supply costs drop.", "supply": -1.5, "demand": 200},
                {"title": "Oil Price Spike", "msg": "Energy costs rise. Inflation pressure increases.", "supply": 2.0, "demand": -100},
                {"title": "Consumer Confidence Boom", "msg": "Households are spending aggressively.", "supply": 0.5, "demand": 800},
                {"title": "Stock Market Correction", "msg": "Wealth effect evaporates. Consumption drops.", "supply": 0, "demand": -800},
                {"title": "Global Trade Slowdown", "msg": "Exports fall. Demand weakens.", "supply": 0.2, "demand": -500},
            ]
            evt = random.choice(events)
            self.state['Supply_Shock'] += evt['supply']
            self.state['Demand_Shock'] += evt['demand']
            self.active_event = evt

    def _record_monthly_data(self, start_state, end_state, steps):
        """Interpolates between start and end state to create monthly history."""
        if steps == 0:
            row = start_state.copy()
            row['Turn'] = self.turn
            row['Month'] = self.month
            self.history.append(row)
            return

        for i in range(1, steps + 1):
            fraction = i / steps
            row = {}
            for key in start_state:
                if isinstance(start_state[key], (int, float)):
                    val = start_state[key] + (end_state[key] - start_state[key]) * fraction
                    # Add cosmetic noise for realism
                    if key in ['Inflation', 'Unemployment', 'GDP_Real']:
                         val += np.random.normal(0, 0.05 * abs(val))
                    row[key] = val
                else:
                    row[key] = end_state[key]
            
            self.month += 1
            row['Turn'] = self.turn
            row['Month'] = self.month
            self.history.append(row)

    def get_history_df(self):
        return pd.DataFrame(self.history)


# ==============================================================================
# MODULE 2: POLITICAL ENGINE
# ==============================================================================
class PoliticalEngine:
    def __init__(self):
        self.approval = 60.0
        self.max_turns = 20

    def calculate_approval(self, economy_state: Dict, history_df: pd.DataFrame) -> float:
        inf = economy_state['Inflation']
        unemp = economy_state['Unemployment']
        
        # Loss Function
        inf_penalty = 1.5 * (inf - CONSTANTS['INFLATION_TARGET'])**2
        unemp_penalty = 2.0 * (unemp - CONSTANTS['NATURAL_UNEMPLOYMENT'])**2
        
        growth_bonus = 0
        # Calculate Year-over-Year Growth for stability
        if len(history_df) > 12:
            prev_gdp = history_df.iloc[-12]['GDP_Real']
            curr_gdp = economy_state['GDP_Real']
            growth = ((curr_gdp - prev_gdp) / prev_gdp) * 100
            growth_bonus = growth * 3

        current_score = 70 - inf_penalty - unemp_penalty + growth_bonus
        self.approval = (0.8 * self.approval) + (0.2 * current_score) # More inertia
        self.approval = max(0, min(100, self.approval))
        
        return self.approval

    def get_advisor_comment(self, state: Dict) -> str:
        inf = state['Inflation']
        unemp = state['Unemployment']
        gap = state['Output_Gap']
        
        if inf > 8.0 and unemp > 7.0:
            return "⚠️ STAGFLATION: Prices rising AND jobs lost. Raising rates kills jobs; spending fuels inflation. Suggest: Structural reform + Careful rate hikes."
        elif gap > 2.5:
            return f"🔥 OVERHEATING (Gap: +{gap:.1f}%): Economy is running too hot. Inflation will spiral if you don't cool it down (Rate hike or Tax hike)."
        elif gap < -2.5:
            return f"❄️ RECESSION (Gap: {gap:.1f}%): Output is far below potential. Stimulate Demand via Gov Spending or Tax cuts."
        elif state['Debt_to_GDP'] > 120:
             return "📉 DEBT CRISIS: Debt > 120% GDP. Markets are panicking. You must reduce the deficit."
        else:
            return "✅ STABLE: Keep maintaining the balance. Watch the Output Gap."

    def check_game_over(self, turn: int) -> Tuple[bool, str]:
        if turn < 4:
            return False, ""
        if self.approval < 20:
            return True, "EJECTED: Approval Rating fell below 20%."
        if turn >= self.max_turns:
            return True, "TERM LIMIT: 5 Years completed."
        return False, ""
    
    def analyze_failure(self, df: pd.DataFrame) -> str:
        avg_inf = df['Inflation'].mean()
        avg_unemp = df['Unemployment'].mean()
        final_debt = df.iloc[-1]['Debt_to_GDP']
        
        reasons = []
        if avg_inf > 5.0: reasons.append(f"❌ **Runaway Inflation ({avg_inf:.1f}%)**: Rates were too low for too long.")
        if avg_unemp > 7.0: reasons.append(f"❌ **Mass Unemployment ({avg_unemp:.1f}%)**: You suffocated the economy.")
        if final_debt > 130.0: reasons.append(f"❌ **Fiscal Collapse ({final_debt:.1f}% Debt)**: You spent money you didn't have.")
            
        return "\n\n".join(reasons) if reasons else "⚠️ **Volatility**: The economy swung too wildly."


# ==============================================================================
# MODULE 3: INTERFACE
# ==============================================================================
class SimulationUI:
    def __init__(self):
        if 'economy' not in st.session_state: st.session_state['economy'] = None
        if 'politics' not in st.session_state: st.session_state['politics'] = None
        if 'game_active' not in st.session_state: st.session_state['game_active'] = False

    def render_sidebar(self):
        st.sidebar.title("🏛️ Oval Office")
        
        if not st.session_state['game_active']:
            scenario = st.sidebar.selectbox("Choose Scenario", ["Standard", "Stagflation (1970s)", "Financial Crisis (2008)"])
            if st.sidebar.button("Start New Term"):
                st.session_state['economy'] = EconomyModel(scenario)
                st.session_state['politics'] = PoliticalEngine()
                st.session_state['game_active'] = True
                st.rerun()
            return None
        
        st.sidebar.markdown("### 🕹️ Policy Levers")
        with st.sidebar.form("policy_form"):
            st.sidebar.markdown("#### Fiscal Policy")
            gov_spending = st.slider("Government Spending ($B)", 2000, 8000, int(st.session_state['economy'].state['Gov_Spending']), 100)
            
            # Tax Rate
            curr_tax_revenue = st.session_state['economy'].state.get('Tax_Revenue', 4000)
            curr_gdp = st.session_state['economy'].state['GDP_Real']
            # Safety check for zero div
            default_tax = int((curr_tax_revenue / curr_gdp * 100)) if curr_gdp > 0 else 20
            
            tax_rate = st.slider("Income Tax Rate (%)", 10, 60, default_tax, 1) / 100.0

            st.sidebar.markdown("#### Monetary Policy")
            interest_rate = st.slider("Fed Funds Rate (%)", 0.0, 20.0, float(st.session_state['economy'].state['Interest_Rate']), 0.25)

            st.sidebar.markdown("#### Structural Reform")
            supply_reform = st.checkbox("Invest in Education/Infra ($100B)")

            if st.form_submit_button("📢 Execute Policy (Next Quarter)"):
                inputs = {'gov_spending': gov_spending, 'tax_rate': tax_rate, 'interest_rate': interest_rate, 'supply_reform': supply_reform}
                self.process_turn(inputs)

        if st.sidebar.button("Resign (Reset Game)"):
            st.session_state['game_active'] = False
            st.rerun()

    def process_turn(self, inputs):
        eco = st.session_state['economy']
        pol = st.session_state['politics']
        eco.step(inputs)
        pol.calculate_approval(eco.state, eco.get_history_df())
        st.rerun()

    def render_dashboard(self):
        if not st.session_state['game_active']:
            st.title("MacroSim 2025 📈")
            st.markdown("### Welcome, Mr./Ms. Chairperson.")
            st.info("Select a scenario in the sidebar to begin your term.")
            return

        eco = st.session_state['economy']
        pol = st.session_state['politics']
        df = eco.get_history_df()
        
        # --- HEADLINES & EVENTS ---
        if eco.active_event:
            st.warning(f"📰 **BREAKING NEWS: {eco.active_event['title']}** — {eco.active_event['msg']}")
        
        # --- TOP METRICS ---
        c1, c2, c3, c4 = st.columns(4)
        curr_inf = eco.state['Inflation']
        curr_unemp = eco.state['Unemployment']
        curr_gdp = eco.state['GDP_Real']
        
        # Get Delta from 3 months ago (Previous Quarter)
        if len(df) > 3:
            prev_row = df.iloc[-4]
            d_inf = curr_inf - prev_row['Inflation']
            d_unemp = curr_unemp - prev_row['Unemployment']
        else:
            d_inf, d_unemp = 0, 0
            
        c1.metric("Inflation", f"{curr_inf:.1f}%", f"{d_inf:+.1f}%", delta_color="inverse")
        c2.metric("Unemployment", f"{curr_unemp:.1f}%", f"{d_unemp:+.1f}%", delta_color="inverse")
        c3.metric("Real GDP", f"${int(curr_gdp):,}B", "Growth")
        c4.metric("Approval", f"{int(pol.approval)}%", f"Turn {eco.turn}/20")
        
        # --- QUARTERLY BRIEFING ---
        with st.expander("📋 Quarterly Briefing (Click to Expand)", expanded=True):
            cols = st.columns(3)
            cols[0].info(f"**Advisor:** {pol.get_advisor_comment(eco.state)}")
            cols[1].metric("Output Gap", f"{eco.state['Output_Gap']:.1f}%", help="Positive=Overheating, Negative=Recession")
            cols[2].metric("Debt-to-GDP", f"{eco.state['Debt_to_GDP']:.1f}%")

        # --- GAME OVER ---
        is_over, reason = pol.check_game_over(eco.turn)
        if is_over:
            st.error(f"GAME OVER: {reason}")
            st.markdown(pol.analyze_failure(df))
            if st.button("Start New Game"):
                st.session_state['game_active'] = False
                st.rerun()
            return

        # --- CHARTS ---
        tab1, tab2, tab3 = st.tabs(["📊 Monthly Trends", "🌀 Phillips Curve", "💰 Debt & Rates"])
        
        with tab1:
            # Use 'Month' instead of 'Turn'
            base = alt.Chart(df).encode(x=alt.X('Month', title="Month (3 per Quarter)"))
            line_inf = base.mark_line(color='red', strokeWidth=3).encode(y='Inflation')
            line_unemp = base.mark_line(color='blue', strokeWidth=3).encode(y='Unemployment')
            st.altair_chart((line_inf + line_unemp).interactive(), use_container_width=True)
            st.caption("Red: Inflation | Blue: Unemployment")

        with tab2:
            chart_pc = alt.Chart(df).mark_circle(size=60).encode(
                x=alt.X('Unemployment', scale=alt.Scale(domain=[0, 15])),
                y=alt.Y('Inflation', scale=alt.Scale(domain=[-2, 15])),
                color=alt.value('purple'),
                tooltip=['Month', 'Inflation', 'Unemployment']
            ).properties(title="Phillips Curve Path")
            line_pc = alt.Chart(df).mark_line(color='gray', strokeDash=[5,5]).encode(
                x='Unemployment', y='Inflation', order='Month'
            )
            st.altair_chart(chart_pc + line_pc, use_container_width=True)

        with tab3:
            col_a, col_b = st.columns(2)
            with col_a:
                chart_debt = alt.Chart(df).mark_area(color='darkred', opacity=0.5).encode(
                    x='Month', y=alt.Y('Debt_to_GDP', scale=alt.Scale(domain=[80, 150]))
                )
                st.altair_chart(chart_debt, use_container_width=True)
            with col_b:
                df['Zero_Line'] = 0
                chart_rate = alt.Chart(df).mark_line(color='green').encode(x='Month', y='Real_Interest_Rate')
                rule = alt.Chart(df).mark_rule(color='black').encode(y='Zero_Line')
                st.altair_chart(chart_rate + rule, use_container_width=True)

if __name__ == "__main__":
    st.set_page_config(page_title="MacroSim 2025", layout="wide")
    app = SimulationUI()
    app.render_sidebar()
    app.render_dashboard()