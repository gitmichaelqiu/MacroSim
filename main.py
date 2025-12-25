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
    'ALPHA': 100,         # IS Curve Sensitivity (Investment)
    'BETA': 0.5,          # Phillips Curve Slope
    'GAMMA': 0.4,         # Okun's Law Coefficient
    'THETA': 500,         # Exchange Rate Sensitivity (Net Export impact)
    'NATURAL_UNEMPLOYMENT': 5.0, 
    'INFLATION_TARGET': 2.0,     
    'POTENTIAL_GROWTH': 2.0,     
    'WORLD_RATE': 4.0,    # Foreign Interest Rate Anchor
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
        self.scenario_name = scenario
        
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
            'Exchange_Rate': 100.0, # Index, 100 = Parity
            'Gov_Spending': 4000,
            'Tax_Revenue': 4000,
            'Net_Exports': 0,
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
        
        # 1. Handle Random Events
        self._trigger_random_event()
        
        # 2. Unpack Inputs
        tax_rate = user_inputs['tax_rate']
        gov_spending = user_inputs['gov_spending']
        nominal_rate = user_inputs['interest_rate']
        
        supply_investment = 100 if user_inputs['supply_reform'] else 0
        gov_spending += supply_investment

        # 3. Monetary Policy & Exchange Rates (Unit 6)
        self.interest_rate_queue.append(nominal_rate)
        effective_nominal_rate = self.interest_rate_queue[0]
        
        # Fisher Equation
        expected_inflation = self.state['Inflation']
        real_interest_rate = effective_nominal_rate - expected_inflation
        
        # Exchange Rate Model (Interest Rate Parity approximation)
        # Higher Rates -> Currency Appreciation -> Higher Index
        rate_differential = nominal_rate - CONSTANTS['WORLD_RATE']
        # Base 100 + 5 points for every 1% differential
        target_exchange_rate = 100 + (rate_differential * 5)
        # Smooth the transition (Currency markets move fast, but not instant)
        new_exchange_rate = (0.3 * self.state['Exchange_Rate']) + (0.7 * target_exchange_rate)

        # Net Exports (NX)
        # Appreciation (High Exchange Rate) -> Lower Exports
        # Sensitivity: 1 point index rise = -$50B NX
        net_exports = -50 * (new_exchange_rate - 100)

        # 4. IS Curve (Demand)
        multiplier = 1 / (1 - CONSTANTS['MPC'] * (1 - tax_rate))
        
        autonomous_consumption = 2500 
        base_investment = 1700
        investment_sensitivity = CONSTANTS['ALPHA'] * real_interest_rate
        
        noise_val = np.random.normal(0, 30)
        total_demand_shock = self.state.get('Demand_Shock', 0) + noise_val
        
        # AD = C + I + G + NX
        base_demand = (autonomous_consumption + 
                       base_investment - investment_sensitivity + 
                       gov_spending + 
                       net_exports +  # <-- Added Open Economy Component
                       total_demand_shock)
                       
        new_real_gdp = base_demand * multiplier

        # 5. Phillips Curve (Supply)
        potential_gdp = self.state['Potential_GDP']
        output_gap_percent = ((new_real_gdp - potential_gdp) / potential_gdp) * 100
        
        total_supply_shock = self.state.get('Supply_Shock', 0) + np.random.normal(0, 0.1)
        
        # Import Prices Pass-through:
        # Strong Currency (High Ex Rate) -> Cheaper Imports -> Lower Inflation
        import_price_effect = -0.1 * (new_exchange_rate - 100)
        
        new_inflation = expected_inflation + (CONSTANTS['BETA'] * output_gap_percent) + total_supply_shock + import_price_effect
        
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
            'Exchange_Rate': new_exchange_rate,
            'Net_Exports': net_exports,
            'Gov_Spending': gov_spending,
            'Tax_Revenue': tax_revenue,
            'Deficit': deficit,
            'Debt_to_GDP': debt_to_gdp
        })
        
        # 8. Generate Monthly Interpolated History
        self._record_monthly_data(prev_state, self.state, 3)

    def _trigger_random_event(self):
        """Randomly injects shocks into the economy."""
        self.state['Demand_Shock'] *= 0.8
        self.state['Supply_Shock'] *= 0.8
        self.active_event = None

        if self.scenario_name == "Sandbox Mode": return 

        if random.random() < 0.25:
            events = [
                {"title": "Tech Breakthrough", "msg": "Productivity soars! Supply costs drop.", "supply": -1.5, "demand": 200},
                {"title": "Oil Price Spike", "msg": "Energy costs rise. Inflation pressure increases.", "supply": 2.0, "demand": -100},
                {"title": "Consumer Confidence Boom", "msg": "Households are spending aggressively.", "supply": 0.5, "demand": 800},
                {"title": "Global Trade Slowdown", "msg": "Foreign partners are buying less. Net Exports fall.", "supply": 0.0, "demand": -1000},
                {"title": "Currency Speculation", "msg": "Investors flee the currency. Import prices rise!", "supply": 1.0, "demand": 200},
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
        if len(history_df) > 12:
            prev_gdp = history_df.iloc[-12]['GDP_Real']
            curr_gdp = economy_state['GDP_Real']
            growth = ((curr_gdp - prev_gdp) / prev_gdp) * 100
            growth_bonus = growth * 3

        current_score = 70 - inf_penalty - unemp_penalty + growth_bonus
        self.approval = (0.8 * self.approval) + (0.2 * current_score) 
        self.approval = max(0, min(100, self.approval))
        
        return self.approval

    def get_advisor_comment(self, state: Dict) -> str:
        inf = state['Inflation']
        unemp = state['Unemployment']
        gap = state['Output_Gap']
        nx = state['Net_Exports']
        
        if inf > 8.0 and unemp > 7.0:
            return "⚠️ STAGFLATION: Prices rising AND jobs lost. Raising rates kills jobs; spending fuels inflation. Suggest: Structural reform."
        elif gap > 2.5:
            return f"🔥 OVERHEATING (Gap: +{gap:.1f}%): Economy is running too hot. Inflation is spiraling."
        elif gap < -2.5:
            return f"❄️ RECESSION (Gap: {gap:.1f}%): Output is far below potential. Stimulate Demand."
        elif nx < -1000:
             return "📉 TRADE DEFICIT ALERT: Your high interest rates are making exports too expensive! The manufacturing sector is hurting."
        elif state['Debt_to_GDP'] > 120:
             return "📉 DEBT CRISIS: Debt > 120% GDP. Reduce the deficit immediately."
        else:
            return "✅ STABLE: Keep maintaining the balance."

    def check_game_over(self, turn: int, scenario: str) -> Tuple[bool, str]:
        if scenario == "Sandbox Mode":
            return False, "" 
        if turn < 4:
            return False, ""
        if self.approval < 20:
            return True, "EJECTED: Approval Rating fell below 20%."
        if turn >= self.max_turns:
            return True, "TERM LIMIT: 5 Years completed."
        return False, ""
    
    def generate_term_report(self, df: pd.DataFrame) -> str:
        avg_inf = df['Inflation'].mean()
        avg_unemp = df['Unemployment'].mean()
        std_inf = df['Inflation'].std()
        
        start_gdp = df.iloc[0]['GDP_Real']
        end_gdp = df.iloc[-1]['GDP_Real']
        total_growth = ((end_gdp - start_gdp) / start_gdp) * 100
        final_debt = df.iloc[-1]['Debt_to_GDP']
        
        grades = {}
        
        inf_dev = abs(avg_inf - 2.0)
        if inf_dev < 1.0 and std_inf < 1.5: grades['Inflation'] = 'A'
        elif inf_dev < 2.0: grades['Inflation'] = 'B'
        elif inf_dev < 4.0: grades['Inflation'] = 'C'
        else: grades['Inflation'] = 'F'
            
        unemp_dev = avg_unemp - 5.0
        if unemp_dev < 0.5 and unemp_dev > -0.5: grades['Employment'] = 'A'
        elif unemp_dev < 1.5: grades['Employment'] = 'B'
        elif unemp_dev < 3.0: grades['Employment'] = 'C'
        else: grades['Employment'] = 'F'
        
        if final_debt < 90: grades['Fiscal'] = 'A'
        elif final_debt < 110: grades['Fiscal'] = 'B'
        elif final_debt < 130: grades['Fiscal'] = 'C'
        else: grades['Fiscal'] = 'D'
        
        report = f"""
        ### 🎓 Official Term Report Card
        
        **Final Grades:**
        - **Price Stability (Inflation):** {grades['Inflation']}
        - **Full Employment:** {grades['Employment']}
        - **Fiscal Responsibility:** {grades['Fiscal']}
        
        **Key Statistics:**
        - **Avg Inflation:** {avg_inf:.1f}% (Target: 2.0%)
        - **Avg Unemployment:** {avg_unemp:.1f}% (Target: 5.0%)
        - **Total Real GDP Growth:** {total_growth:.1f}% over term
        - **Ending Debt-to-GDP:** {final_debt:.1f}%
        """
        return report

# ==============================================================================
# MODULE 3: VISUALIZATION HELPERS (THEORY LAB)
# ==============================================================================
def get_adas_chart(current_gdp, potential_gdp, price_level, gov_spending_delta=0):
    x = np.linspace(potential_gdp * 0.8, potential_gdp * 1.2, 100)
    lras_df = pd.DataFrame({'GDP': [potential_gdp, potential_gdp], 'Price': [80, 120], 'Type': ['LRAS', 'LRAS']})
    
    slope_sras = 0.002
    sras_y = price_level + slope_sras * (x - potential_gdp)
    sras_df = pd.DataFrame({'GDP': x, 'Price': sras_y, 'Type': ['SRAS']*100})
    
    slope_ad = 0.003
    base_intercept = price_level + slope_ad * current_gdp
    shift_y = gov_spending_delta * 3.0 
    ad_y = base_intercept - slope_ad * (x - shift_y)
    ad_df = pd.DataFrame({'GDP': x, 'Price': ad_y, 'Type': ['AD (Aggregate Demand)']*100})
    
    final_df = pd.concat([lras_df, sras_df, ad_df])
    
    chart = alt.Chart(final_df).mark_line().encode(
        x=alt.X('GDP', scale=alt.Scale(domain=[potential_gdp*0.8, potential_gdp*1.2])),
        y=alt.Y('Price', scale=alt.Scale(domain=[90, 110])),
        color='Type'
    ).properties(title="AD-AS Model")
    return chart

def get_forex_chart(exchange_rate, net_exports):
    """Generates simple Bar chart for Trade Balance."""
    df = pd.DataFrame({
        'Metric': ['Net Exports', 'Exchange Rate Idx'],
        'Value': [net_exports, exchange_rate]
    })
    return alt.Chart(df).mark_bar().encode(
        x='Metric', y='Value', color='Metric'
    ).properties(title="External Sector")


# ==============================================================================
# MODULE 4: INTERFACE
# ==============================================================================
class SimulationUI:
    def __init__(self):
        if 'economy' not in st.session_state: st.session_state['economy'] = None
        if 'politics' not in st.session_state: st.session_state['politics'] = None
        if 'game_active' not in st.session_state: st.session_state['game_active'] = False

    def render_sidebar(self):
        st.sidebar.title("🏛️ Oval Office")
        
        if not st.session_state['game_active']:
            scenario = st.sidebar.selectbox("Choose Scenario", ["Standard", "Sandbox Mode", "Stagflation (1970s)", "Financial Crisis (2008)"])
            if st.sidebar.button("Start New Term"):
                st.session_state['economy'] = EconomyModel(scenario)
                st.session_state['politics'] = PoliticalEngine()
                st.session_state['game_active'] = True
                st.rerun()
            return None
        
        st.sidebar.markdown("### 🕹️ Policy Levers")
        
        with st.sidebar.form("policy_form"):
            st.sidebar.markdown("#### Fiscal Policy")
            current_g = int(st.session_state['economy'].state['Gov_Spending'])
            gov_spending = st.slider("Government Spending ($B)", 2000, 8000, current_g, 100)
            
            curr_tax_revenue = st.session_state['economy'].state.get('Tax_Revenue', 4000)
            curr_gdp = st.session_state['economy'].state['GDP_Real']
            default_tax = int((curr_tax_revenue / curr_gdp * 100)) if curr_gdp > 0 else 20
            tax_rate = st.slider("Income Tax Rate (%)", 10, 60, default_tax, 1) / 100.0

            st.sidebar.markdown("#### Monetary Policy")
            interest_rate = st.slider("Fed Funds Rate (%)", 0.0, 20.0, float(st.session_state['economy'].state['Interest_Rate']), 0.25)

            st.sidebar.markdown("#### Structural Reform")
            supply_reform = st.checkbox("Invest in Education/Infra ($100B)")

            if st.form_submit_button("📢 Execute Policy (Next Quarter)"):
                inputs = {'gov_spending': gov_spending, 'tax_rate': tax_rate, 'interest_rate': interest_rate, 'supply_reform': supply_reform}
                self.process_turn(inputs)

        self.current_inputs = {'G': gov_spending, 'T': tax_rate, 'i': interest_rate}
        
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
        
        if eco.active_event:
            st.warning(f"📰 **BREAKING NEWS: {eco.active_event['title']}** — {eco.active_event['msg']}")
        
        # --- METRICS ---
        c1, c2, c3, c4, c5 = st.columns(5) # Added 5th col for Exchange Rate
        curr_inf = eco.state['Inflation']
        curr_unemp = eco.state['Unemployment']
        curr_gdp = eco.state['GDP_Real']
        curr_xr = eco.state['Exchange_Rate']
        
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
        c5.metric("Exchange Rate", f"{curr_xr:.1f}", help=">100 = Strong Currency (Hurts Exports)")
        
        with st.expander("📋 Quarterly Briefing (Click to Expand)", expanded=True):
            cols = st.columns(3)
            cols[0].info(f"**Advisor:** {pol.get_advisor_comment(eco.state)}")
            cols[1].metric("Output Gap", f"{eco.state['Output_Gap']:.1f}%")
            cols[2].metric("Net Exports", f"${int(eco.state['Net_Exports'])}B")

        # --- GAME OVER / EXPORT ---
        is_over, reason = pol.check_game_over(eco.turn, eco.scenario_name)
        if is_over:
            if "EJECTED" in reason: st.error(f"GAME OVER: {reason}")
            else: st.success(f"CONGRATULATIONS: {reason}")
            st.markdown(pol.generate_term_report(df))
            
            # DATA EXPORT
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Term Data (CSV)", csv, "macro_sim_report.csv", "text/csv")
            
            if st.button("Start New Game"):
                st.session_state['game_active'] = False
                st.rerun()
            return

        # --- CHARTS ---
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Main Trends", "📉 Theory Lab", "🌍 Trade & Forex", "💰 Debt & Rates"])
        
        with tab1:
            base = alt.Chart(df).encode(x=alt.X('Month', title="Month"))
            line_inf = base.mark_line(color='red', strokeWidth=3).encode(y='Inflation')
            line_unemp = base.mark_line(color='blue', strokeWidth=3).encode(y='Unemployment')
            st.altair_chart((line_inf + line_unemp).interactive(), use_container_width=True)

        with tab2:
            st.markdown("#### Real-Time Structural Analysis")
            col_ad, col_money = st.columns(2)
            
            # AD-AS
            delta_g = self.current_inputs['G'] - eco.state['Gov_Spending']
            adas_chart = get_adas_chart(eco.state['GDP_Real'], eco.state['Potential_GDP'], eco.state['Price_Level'], delta_g)
            col_ad.altair_chart(adas_chart, use_container_width=True)
            col_ad.caption("Impact of Government Spending on Aggregate Demand.")
            
            # Trade visualization in Theory Lab
            st.markdown(f"**Implied Exchange Rate Impact:** Rate {self.current_inputs['i']}% vs World 4% -> Net Export Impact: ${int(-50 * ((100 + (self.current_inputs['i']-4)*5) - 100))}B")

        with tab3:
            col_x, col_y = st.columns(2)
            with col_x:
                base = alt.Chart(df).encode(x='Month')
                line_nx = base.mark_area(color='teal', opacity=0.6).encode(y=alt.Y('Net_Exports', title='Net Exports ($B)'))
                st.altair_chart(line_nx, use_container_width=True)
            with col_y:
                base = alt.Chart(df).encode(x='Month')
                line_xr = base.mark_line(color='orange').encode(y=alt.Y('Exchange_Rate', title='Exchange Rate Index (100=Parity)'))
                st.altair_chart(line_xr, use_container_width=True)

        with tab4:
            col_a, col_b = st.columns(2)
            with col_a:
                chart_debt = alt.Chart(df).mark_area(color='darkred', opacity=0.5).encode(x='Month', y=alt.Y('Debt_to_GDP', scale=alt.Scale(domain=[80, 150])))
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