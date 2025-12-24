import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from collections import deque
from typing import Dict, List, Tuple

# ==============================================================================
# CONFIGURATION & CONSTANTS
# ==============================================================================
CONSTANTS = {
    'MPC': 0.75,          # Marginal Propensity to Consume
    'ALPHA': 1500,        # IS Curve Sensitivity (Interest Rate Sensitivity)
    'BETA': 0.5,          # Phillips Curve Slope (Inflation Sensitivity)
    'GAMMA': 0.4,         # Okun's Law Coefficient
    'NATURAL_UNEMPLOYMENT': 5.0, # NAIRU (%)
    'INFLATION_TARGET': 2.0,     # Target Inflation (%)
    'POTENTIAL_GROWTH': 2.0,     # Annual Potential GDP Growth (%)
    'DEPRECIATION': 0.05,        # Capital Depreciation
}

# ==============================================================================
# MODULE 1: ECONOMIC ENGINE (MODEL)
# Handles the "Physics" of the economy (IS-LM, Phillips Curve, Solow Growth)
# ==============================================================================
class EconomyModel:
    def __init__(self, scenario: str = "Standard"):
        self.turn = 0
        self.history = []
        self._initialize_state(scenario)
        
        # Policy Lag Mechanism: Stores past 2 quarters of interest rates
        # Default starting rate is 4.0%
        self.interest_rate_queue = deque([4.0, 4.0], maxlen=2)

    def _initialize_state(self, scenario: str):
        """Initialize the economy based on the selected scenario."""
        # Baseline Defaults
        self.state = {
            'GDP_Nominal': 20000,   # Billions
            'GDP_Real': 20000,      # Billions
            'Potential_GDP': 20000, # Billions
            'Price_Level': 100.0,
            'Inflation': 2.0,       # %
            'Unemployment': 5.0,    # %
            'Interest_Rate': 4.0,   # % (Fed Funds)
            'Real_Interest_Rate': 2.0, # % (Real)
            'Gov_Spending': 4000,   # Billions
            'Tax_Revenue': 4000,    # Billions
            'Deficit': 0,           # Billions
            'Debt': 20000,          # Billions
            'Debt_to_GDP': 100.0,   # %
            'Supply_Shock': 0.0,    # Random noise
            'Demand_Shock': 0.0,    # Random noise
        }

        # Scenario Overrides
        if scenario == "Stagflation (1970s)":
            self.state['Inflation'] = 9.0
            self.state['Unemployment'] = 7.5
            self.state['Supply_Shock'] = 3.0 # Persistent shock
            
        elif scenario == "Financial Crisis (2008)":
            self.state['Inflation'] = 1.0
            self.state['Unemployment'] = 8.0
            self.state['Demand_Shock'] = -1500 # Massive demand drop
            self.state['Interest_Rate'] = 0.5

        # Record initial state
        self._record_history()

    def step(self, user_inputs: Dict):
        """
        Advances the economy by one turn (Quarter).
        """
        self.turn += 1
        
        # 1. Unpack Inputs
        tax_rate = user_inputs['tax_rate']
        gov_spending = user_inputs['gov_spending']
        nominal_rate = user_inputs['interest_rate']
        
        supply_investment = 100 if user_inputs['supply_reform'] else 0
        gov_spending += supply_investment

        # 2. Monetary Policy Transmission (Lagged)
        self.interest_rate_queue.append(nominal_rate)
        effective_nominal_rate = self.interest_rate_queue[0] # Pop the oldest
        
        # Fisher Equation: r = i - pi^e
        expected_inflation = self.state['Inflation']
        real_interest_rate = effective_nominal_rate - expected_inflation

        # 3. The IS Curve (Aggregate Demand)
        # Multiplier: 1 / (1 - MPC*(1-t))
        multiplier = 1 / (1 - CONSTANTS['MPC'] * (1 - tax_rate))
        
        # Autonomous Demand
        autonomous_consumption = 4000 
        investment_sensitivity = CONSTANTS['ALPHA'] * real_interest_rate
        
        shock_val = np.random.normal(0, 50)
        scenario_demand_shock = self.state.get('Demand_Shock', 0)
        
        base_demand = autonomous_consumption + 3000 - investment_sensitivity + gov_spending + scenario_demand_shock + shock_val
        new_real_gdp = base_demand * multiplier

        # 4. The Phillips Curve (Aggregate Supply)
        potential_gdp = self.state['Potential_GDP']
        output_gap_percent = ((new_real_gdp - potential_gdp) / potential_gdp) * 100
        
        scenario_supply_shock = self.state.get('Supply_Shock', 0)
        random_supply_noise = np.random.normal(0, 0.2)
        
        new_inflation = expected_inflation + (CONSTANTS['BETA'] * output_gap_percent) + scenario_supply_shock + random_supply_noise
        
        # 5. Okun's Law (Unemployment)
        new_unemployment = CONSTANTS['NATURAL_UNEMPLOYMENT'] - (CONSTANTS['GAMMA'] * output_gap_percent)
        new_unemployment = max(0.5, new_unemployment) # Floor at 0.5%

        # 6. Long Run Updates
        growth_rate = CONSTANTS['POTENTIAL_GROWTH'] / 4 # Quarterly
        if user_inputs['supply_reform']:
            growth_rate += 0.2 
            
        self.state['Potential_GDP'] *= (1 + growth_rate/100)
        
        # Debt & Deficits
        tax_revenue = new_real_gdp * tax_rate
        deficit = gov_spending - tax_revenue
        self.state['Debt'] += deficit
        
        # Price Level
        self.state['Price_Level'] *= (1 + new_inflation/100)
        
        # Ratios
        new_nominal_gdp = new_real_gdp * (self.state['Price_Level'] / 100)
        debt_to_gdp = (self.state['Debt'] / new_nominal_gdp) * 100

        # Update State
        self.state.update({
            'GDP_Real': new_real_gdp,
            'GDP_Nominal': new_nominal_gdp,
            'Inflation': new_inflation,
            'Unemployment': new_unemployment,
            'Interest_Rate': nominal_rate,
            'Real_Interest_Rate': real_interest_rate,
            'Gov_Spending': gov_spending,
            'Tax_Revenue': tax_revenue,
            'Deficit': deficit,
            'Debt_to_GDP': debt_to_gdp
        })
        
        self._record_history()

    def _record_history(self):
        snapshot = self.state.copy()
        snapshot['Turn'] = self.turn
        self.history.append(snapshot)

    def get_history_df(self):
        return pd.DataFrame(self.history)


# ==============================================================================
# MODULE 2: POLITICAL ENGINE (GAME LOGIC)
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
        if len(history_df) > 1:
            prev_gdp = history_df.iloc[-2]['GDP_Real']
            curr_gdp = economy_state['GDP_Real']
            growth = ((curr_gdp - prev_gdp) / prev_gdp) * 100
            growth_bonus = growth * 5

        current_score = 70 - inf_penalty - unemp_penalty + growth_bonus
        self.approval = (0.7 * self.approval) + (0.3 * current_score)
        self.approval = max(0, min(100, self.approval))
        
        return self.approval

    def get_advisor_comment(self, state: Dict) -> str:
        inf = state['Inflation']
        unemp = state['Unemployment']
        r_rate = state['Real_Interest_Rate']
        
        if inf > 8.0 and unemp > 7.0:
            return "⚠️ STAGFLATION: Prices rising AND jobs lost. Raising rates kills jobs; spending fuels inflation. Suggest: Structural reform + Careful rate hikes."
        elif inf > 6.0:
            msg = "🔥 OVERHEATING: Economy is running too hot."
            if r_rate < 0:
                msg += " Your Real Interest Rate is NEGATIVE, effectively subsidizing borrowing. Raise rates!"
            else:
                msg += " Consider raising taxes or cutting spending."
            return msg
        elif unemp > 8.0:
            return "❄️ RECESSION: High unemployment. Stimulate demand! If rates are near zero, use Fiscal Policy (G up, T down)."
        elif state['Debt_to_GDP'] > 120:
             return "📉 DEBT CRISIS: Debt-to-GDP is critical (>120%). Markets may lose confidence. Try to run a primary surplus."
        else:
            return "✅ STABLE: Keep maintaining the balance. Watch the Real Interest Rate."

    def check_game_over(self, turn: int) -> Tuple[bool, str]:
        if turn < 4:
            return False, ""

        if self.approval < 20:
            return True, "EJECTED: Approval Rating fell below 20%."
        if turn >= self.max_turns:
            return True, "TERM LIMIT: 5 Years completed."
        return False, ""

    def analyze_failure(self, df: pd.DataFrame) -> str:
        """Generates a post-mortem report on why the player failed."""
        avg_inf = df['Inflation'].mean()
        avg_unemp = df['Unemployment'].mean()
        final_debt = df.iloc[-1]['Debt_to_GDP']
        
        reasons = []
        
        if avg_inf > 5.0:
            reasons.append(f"❌ **Runaway Inflation**: Averaged {avg_inf:.1f}%. You likely kept interest rates too low or spent too much when the economy was already at capacity.")
        
        if avg_unemp > 7.0:
            reasons.append(f"❌ **Mass Unemployment**: Averaged {avg_unemp:.1f}%. You may have tightened policy too aggressively or failed to stimulate during a downturn.")
            
        if final_debt > 130.0:
            reasons.append(f"❌ **Fiscal Irresponsibility**: Debt hit {final_debt:.1f}% of GDP. This creates long-term drag.")
            
        if not reasons:
            return "⚠️ **Volatility**: The economy swung too wildly between boom and bust. Stability is key."
            
        return "\n\n".join(reasons)


# ==============================================================================
# MODULE 3: INTERFACE (VIEW/CONTROLLER)
# ==============================================================================
class SimulationUI:
    def __init__(self):
        if 'economy' not in st.session_state:
            st.session_state['economy'] = None
        if 'politics' not in st.session_state:
            st.session_state['politics'] = None
        if 'game_active' not in st.session_state:
            st.session_state['game_active'] = False

    def render_sidebar(self):
        st.sidebar.title("🏛️ Oval Office")
        
        if not st.session_state['game_active']:
            scenario = st.sidebar.selectbox(
                "Choose Scenario", 
                ["Standard", "Stagflation (1970s)", "Financial Crisis (2008)"]
            )
            if st.sidebar.button("Start New Term"):
                st.session_state['economy'] = EconomyModel(scenario)
                st.session_state['politics'] = PoliticalEngine()
                st.session_state['game_active'] = True
                st.rerun()
            return None
        
        st.sidebar.markdown("### 🕹️ Policy Levers")
        
        with st.sidebar.form("policy_form"):
            st.sidebar.markdown("#### Fiscal Policy")
            gov_spending = st.slider(
                "Government Spending ($B)", 
                min_value=2000, max_value=8000, 
                value=int(st.session_state['economy'].state['Gov_Spending']),
                step=100,
                help="Directly increases Aggregate Demand."
            )
            
            tax_rate = st.slider(
                "Income Tax Rate (%)", 
                min_value=10, max_value=60, 
                value=int(st.session_state['economy'].state.get('Tax_Revenue', 4000) / st.session_state['economy'].state['GDP_Real'] * 100), # Approx start
                step=1
            ) / 100.0

            st.sidebar.markdown("#### Monetary Policy")
            interest_rate = st.slider(
                "Fed Funds Rate (%)",
                min_value=0.0, max_value=20.0,
                value=float(st.session_state['economy'].state['Interest_Rate']),
                step=0.25,
                help="Lagged effect: 6 months."
            )

            st.sidebar.markdown("#### Structural Reform")
            supply_reform = st.checkbox("Invest in Education/Infra ($100B)")

            submitted = st.form_submit_button("📢 Execute Policy (Next Quarter)")
            
            if submitted:
                inputs = {
                    'gov_spending': gov_spending,
                    'tax_rate': tax_rate,
                    'interest_rate': interest_rate,
                    'supply_reform': supply_reform
                }
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
        
        # --- TOP LEVEL METRICS ---
        
        # Row 1: The "Big Three" + Approval
        c1, c2, c3, c4 = st.columns(4)
        curr_inf = eco.state['Inflation']
        curr_unemp = eco.state['Unemployment']
        curr_gdp = eco.state['GDP_Real']
        curr_app = pol.approval
        
        prev_inf = df.iloc[-2]['Inflation'] if len(df) > 1 else curr_inf
        prev_unemp = df.iloc[-2]['Unemployment'] if len(df) > 1 else curr_unemp
        
        c1.metric("Inflation", f"{curr_inf:.1f}%", f"{(curr_inf-prev_inf):.1f}%", delta_color="inverse")
        c2.metric("Unemployment", f"{curr_unemp:.1f}%", f"{(curr_unemp-prev_unemp):.1f}%", delta_color="inverse")
        c3.metric("Real GDP", f"${int(curr_gdp):,}B", "Growth")
        c4.metric("Approval", f"{int(curr_app)}%", f"Turn {eco.turn}/20")
        
        # Row 2: Advanced Indicators (Requested by User)
        st.markdown("---")
        c5, c6, c7, c8 = st.columns(4)
        
        real_rate = eco.state['Real_Interest_Rate']
        debt_gdp = eco.state['Debt_to_GDP']
        deficit = eco.state['Deficit']
        
        c5.metric("Fed Funds Rate", f"{eco.state['Interest_Rate']}%")
        c6.metric("Real Interest Rate", f"{real_rate:.1f}%", help="Nominal Rate - Inflation. If negative, policy is loose.")
        c7.metric("Debt-to-GDP", f"{debt_gdp:.1f}%", help="Sustainable below 100%. Critical > 120%.")
        c8.metric("Qtr Deficit", f"${int(deficit)}B", help="Gov Spending - Tax Revenue")

        # --- ADVISOR ---
        st.info(f"**Advisor:** {pol.get_advisor_comment(eco.state)}")

        # --- GAME OVER LOGIC ---
        is_over, reason = pol.check_game_over(eco.turn)
        if is_over:
            st.error(f"GAME OVER: {reason}")
            
            st.markdown("### 📋 Post-Mortem Analysis")
            st.markdown(pol.analyze_failure(df))
            
            with st.expander("Review Full Term Data"):
                st.dataframe(df)
            
            if st.button("Start New Game"):
                st.session_state['game_active'] = False
                st.rerun()
            return

        # --- CHARTS ---
        tab1, tab2, tab3 = st.tabs(["📊 Main Indicators", "🌀 Phillips Curve", "💰 Debt & Rates"])
        
        with tab1:
            base = alt.Chart(df).encode(x='Turn')
            line_inf = base.mark_line(color='red').encode(y='Inflation')
            line_unemp = base.mark_line(color='blue').encode(y='Unemployment')
            st.altair_chart((line_inf + line_unemp).interactive(), use_container_width=True)
            st.caption("Red: Inflation | Blue: Unemployment")

        with tab2:
            chart_pc = alt.Chart(df).mark_circle(size=60).encode(
                x=alt.X('Unemployment', scale=alt.Scale(domain=[0, 15])),
                y=alt.Y('Inflation', scale=alt.Scale(domain=[-2, 15])),
                color=alt.value('purple'),
                tooltip=['Turn', 'Inflation', 'Unemployment']
            ).properties(title="Phillips Curve Path")
            line_pc = alt.Chart(df).mark_line(color='gray', strokeDash=[5,5]).encode(
                x='Unemployment', y='Inflation', order='Turn'
            )
            st.altair_chart(chart_pc + line_pc, use_container_width=True)

        with tab3:
            # Debt Ratio & Real Rate
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Debt-to-GDP (%)**")
                chart_debt = alt.Chart(df).mark_area(color='darkred', opacity=0.5).encode(
                    x='Turn', y=alt.Y('Debt_to_GDP', scale=alt.Scale(domain=[80, 150]))
                )
                st.altair_chart(chart_debt, use_container_width=True)
            with col_b:
                st.markdown("**Real Interest Rate (%)**")
                df['Zero_Line'] = 0
                chart_rate = alt.Chart(df).mark_line(color='green').encode(
                    x='Turn', y='Real_Interest_Rate'
                )
                rule = alt.Chart(df).mark_rule(color='black').encode(y='Zero_Line')
                st.altair_chart(chart_rate + rule, use_container_width=True)

if __name__ == "__main__":
    st.set_page_config(page_title="MacroSim 2025", layout="wide")
    app = SimulationUI()
    app.render_sidebar()
    app.render_dashboard()