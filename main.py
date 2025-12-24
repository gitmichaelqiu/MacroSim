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
            'Gov_Spending': 4000,   # Billions
            'Tax_Revenue': 4000,    # Billions
            'Debt': 20000,          # Billions
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
        
        Logic Flow:
        1. Process Lags (Interest Rate)
        2. Fiscal Multiplier (IS Curve)
        3. Output Gap Calculation
        4. Phillips Curve (Inflation)
        5. Okun's Law (Unemployment)
        6. Debt & Long Run updates
        """
        self.turn += 1
        
        # 1. Unpack Inputs
        # ----------------
        # Inputs: Tax Rate (0-1), Gov Spending ($B), Fed Funds Rate (%)
        tax_rate = user_inputs['tax_rate']
        gov_spending = user_inputs['gov_spending']
        nominal_rate = user_inputs['interest_rate']
        
        # Supply Side Investments (Cost now, growth later)
        supply_investment = 100 if user_inputs['supply_reform'] else 0
        gov_spending += supply_investment

        # 2. Monetary Policy Transmission (Lagged)
        # ----------------
        # We use the rate set 2 turns ago to determine effective real rate today.
        # This simulates the "Fool in the Shower" problem.
        self.interest_rate_queue.append(nominal_rate)
        effective_nominal_rate = self.interest_rate_queue[0] # Pop the oldest
        
        # Real Interest Rate = Nominal - Expected Inflation (Adaptive)
        # Fisher Equation
        expected_inflation = self.state['Inflation']
        real_interest_rate = effective_nominal_rate - expected_inflation

        # 3. The IS Curve (Aggregate Demand)
        # ----------------
        # Y = Multiplier * (Autonomous Consumption + Investment + G + NX)
        # Simplified: Investment depends on Real Interest Rate (alpha)
        
        # Calculate Multiplier: 1 / (1 - MPC*(1-t))
        multiplier = 1 / (1 - CONSTANTS['MPC'] * (1 - tax_rate))
        
        # Autonomous Demand Components
        # RECALIBRATED: Increased to 4000 to balance equation at G=4000, Y=20000
        autonomous_consumption = 4000 
        investment_sensitivity = CONSTANTS['ALPHA'] * real_interest_rate
        
        # Stochastic Shocks (Randomness)
        shock_val = np.random.normal(0, 50) # Standard volatility
        scenario_demand_shock = self.state.get('Demand_Shock', 0)
        
        # Aggregate Demand Calculation
        # AD = (C0 + I0 - alpha*r + G) * Multiplier
        # Note: '3000' is the base Investment constant.
        base_demand = autonomous_consumption + 3000 - investment_sensitivity + gov_spending + scenario_demand_shock + shock_val
        new_real_gdp = base_demand * multiplier

        # 4. The Phillips Curve (Aggregate Supply)
        # ----------------
        # Inflation = Expected Inflation + Beta * Output_Gap + Supply_Shock
        
        potential_gdp = self.state['Potential_GDP']
        output_gap_percent = ((new_real_gdp - potential_gdp) / potential_gdp) * 100
        
        scenario_supply_shock = self.state.get('Supply_Shock', 0)
        random_supply_noise = np.random.normal(0, 0.2)
        
        new_inflation = expected_inflation + (CONSTANTS['BETA'] * output_gap_percent) + scenario_supply_shock + random_supply_noise
        
        # 5. Okun's Law (Unemployment)
        # ----------------
        # u = u* - Gamma * Output_Gap
        new_unemployment = CONSTANTS['NATURAL_UNEMPLOYMENT'] - (CONSTANTS['GAMMA'] * output_gap_percent)
        new_unemployment = max(0.5, new_unemployment) # Floor at 0.5%

        # 6. Long Run Updates (Solow & Debt)
        # ----------------
        # Update Potential GDP (Growth Trend)
        growth_rate = CONSTANTS['POTENTIAL_GROWTH'] / 4 # Quarterly
        if user_inputs['supply_reform']:
            growth_rate += 0.2 # Boost from reform
            
        self.state['Potential_GDP'] *= (1 + growth_rate/100)
        
        # Update Debt
        tax_revenue = new_real_gdp * tax_rate
        deficit = gov_spending - tax_revenue
        self.state['Debt'] += deficit
        
        # Update Price Level
        self.state['Price_Level'] *= (1 + new_inflation/100)

        # Update State Dict
        self.state.update({
            'GDP_Real': new_real_gdp,
            'GDP_Nominal': new_real_gdp * (self.state['Price_Level'] / 100),
            'Inflation': new_inflation,
            'Unemployment': new_unemployment,
            'Interest_Rate': nominal_rate,
            'Gov_Spending': gov_spending,
            'Tax_Revenue': tax_revenue,
        })
        
        self._record_history()

    def _record_history(self):
        """Snapshots current state to history list."""
        snapshot = self.state.copy()
        snapshot['Turn'] = self.turn
        self.history.append(snapshot)

    def get_history_df(self):
        return pd.DataFrame(self.history)


# ==============================================================================
# MODULE 2: POLITICAL ENGINE (GAME LOGIC)
# Handles Approval Ratings, Advisors, and Win/Loss States
# ==============================================================================
class PoliticalEngine:
    def __init__(self):
        self.approval = 60.0 # Started higher to give buffer
        self.max_turns = 20 # 5 Years (Quarters)

    def calculate_approval(self, economy_state: Dict, history_df: pd.DataFrame) -> float:
        """
        Calculates approval rating based on a Loss Function.
        Voters hate Inflation and Unemployment quadratically.
        """
        inf = economy_state['Inflation']
        unemp = economy_state['Unemployment']
        
        # Loss Function: L = w1(pi - pi*)^2 + w2(u - u*)^2
        # Voters forgive low inflation, but hate high inflation.
        inf_penalty = 1.5 * (inf - CONSTANTS['INFLATION_TARGET'])**2
        unemp_penalty = 2.0 * (unemp - CONSTANTS['NATURAL_UNEMPLOYMENT'])**2
        
        # GDP Growth Bonus (Retrospective)
        growth_bonus = 0
        if len(history_df) > 1:
            prev_gdp = history_df.iloc[-2]['GDP_Real']
            curr_gdp = economy_state['GDP_Real']
            growth = ((curr_gdp - prev_gdp) / prev_gdp) * 100
            growth_bonus = growth * 5 # 1% growth = +5 approval

        # Base Approval Logic
        current_score = 70 - inf_penalty - unemp_penalty + growth_bonus
        
        # Smoothing (Approval doesn't jump instantly, it drifts)
        self.approval = (0.7 * self.approval) + (0.3 * current_score)
        self.approval = max(0, min(100, self.approval)) # Clamp 0-100
        
        return self.approval

    def get_advisor_comment(self, state: Dict) -> str:
        """Generates feedback based on economic conditions."""
        inf = state['Inflation']
        unemp = state['Unemployment']
        
        if inf > 8.0 and unemp > 7.0:
            return "⚠️ STAGFLATION ALERT: We are in a crisis! Prices are rising while jobs are scarce. Standard demand-side stimulus (raising G) might just make inflation worse. Consider supply-side reforms or biting the bullet on interest rates."
        elif inf > 6.0:
            return "🔥 OVERHEATING: Inflation is eating into purchasing power. The economy is running too hot. Consider raising rates or cutting spending to cool Aggregate Demand."
        elif unemp > 8.0:
            return "❄️ RECESSION: Unemployment is dangerously high. Millions are out of work. We need to stimulate Demand immediately. Cut taxes or boost spending!"
        elif state['Debt'] > 40000:
             return "📉 DEBT WARNING: Our national debt is exploding. Bond markets are getting nervous. Be careful with excessive deficits."
        else:
            return "✅ STABLE: The economy is tracking well. Keep maintaining the balance between growth and stability."

    def check_game_over(self, turn: int) -> Tuple[bool, str]:
        # Honeymoon period: Can't lose in first year (4 turns)
        if turn < 4:
            return False, ""

        if self.approval < 20:
            return True, "You have been voted out of office due to historically low approval ratings."
        if turn >= self.max_turns:
            return True, "Term limit reached. History will judge your administration."
        return False, ""


# ==============================================================================
# MODULE 3: INTERFACE (VIEW/CONTROLLER)
# Streamlit UI
# ==============================================================================
class SimulationUI:
    def __init__(self):
        # Initialize Session State
        if 'economy' not in st.session_state:
            st.session_state['economy'] = None
        if 'politics' not in st.session_state:
            st.session_state['politics'] = None
        if 'game_active' not in st.session_state:
            st.session_state['game_active'] = False

    def render_sidebar(self):
        st.sidebar.title("🏛️ Oval Office")
        
        # Scenario Selector (Only active if game not started)
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
        
        # Game Controls
        st.sidebar.markdown("### 🕹️ Policy Levers")
        
        with st.sidebar.form("policy_form"):
            # Fiscal Policy
            st.sidebar.markdown("#### Fiscal Policy")
            gov_spending = st.slider(
                "Government Spending ($B)", 
                min_value=2000, max_value=8000, 
                value=int(st.session_state['economy'].state['Gov_Spending']),
                step=100,
                help="Shifts Aggregate Demand (AD) right. Multiplier effect applies."
            )
            
            tax_rate = st.slider(
                "Income Tax Rate (%)", 
                min_value=10, max_value=60, 
                value=20,
                step=1,
                help="Higher taxes reduce consumption and cool the economy."
            ) / 100.0

            # Monetary Policy
            st.sidebar.markdown("#### Monetary Policy")
            interest_rate = st.slider(
                "Fed Funds Rate (%)",
                min_value=0.0, max_value=20.0,
                value=float(st.session_state['economy'].state['Interest_Rate']),
                step=0.25,
                help="Interest rate lag: Takes 2 turns (6 months) to fully impact Investment."
            )

            # Supply Side
            st.sidebar.markdown("#### Structural Reform")
            supply_reform = st.checkbox(
                "Invest in Education/Infra ($100B)",
                help="Costs money now but increases Potential GDP growth in the long run."
            )

            submitted = st.form_submit_button("📢 Execute Policy (Next Quarter)")
            
            if submitted:
                inputs = {
                    'gov_spending': gov_spending,
                    'tax_rate': tax_rate,
                    'interest_rate': interest_rate,
                    'supply_reform': supply_reform
                }
                self.process_turn(inputs)

        # Reset Button
        if st.sidebar.button("Resign (Reset Game)"):
            st.session_state['game_active'] = False
            st.rerun()

    def process_turn(self, inputs):
        """Update Model and Political State"""
        eco = st.session_state['economy']
        pol = st.session_state['politics']
        
        eco.step(inputs)
        pol.calculate_approval(eco.state, eco.get_history_df())
        
        st.rerun()

    def render_dashboard(self):
        if not st.session_state['game_active']:
            st.title("MacroSim 2025 📈")
            st.markdown("""
            Welcome to **MacroSim**, an interactive economic simulator for AP/A-Level Economics.
            
            **Your Goal:** Manage the economy for a 5-year term (20 Quarters).
            **Winning Conditions:** High Approval Rating, Low Misery Index.
            
            ### Concepts Simulated:
            * **The Keynesian Multiplier**: Watch how $\Delta G$ impacts $Y$.
            * **Time Lags**: Monetary policy takes time to work.
            * **The Phillips Curve**: The trade-off between Inflation and Unemployment.
            
            *Select a Scenario in the sidebar to begin.*
            """)
            return

        eco = st.session_state['economy']
        pol = st.session_state['politics']
        df = eco.get_history_df()
        
        # 1. KPI Row
        # ----------
        col1, col2, col3, col4 = st.columns(4)
        
        # Format metrics
        curr_inf = eco.state['Inflation']
        curr_unemp = eco.state['Unemployment']
        curr_gdp = eco.state['GDP_Real']
        curr_app = pol.approval
        
        # Deltas
        prev_inf = df.iloc[-2]['Inflation'] if len(df) > 1 else curr_inf
        prev_unemp = df.iloc[-2]['Unemployment'] if len(df) > 1 else curr_unemp
        
        col1.metric("Inflation", f"{curr_inf:.1f}%", f"{(curr_inf-prev_inf):.1f}%", delta_color="inverse")
        col2.metric("Unemployment", f"{curr_unemp:.1f}%", f"{(curr_unemp-prev_unemp):.1f}%", delta_color="inverse")
        col3.metric("Real GDP", f"${int(curr_gdp)}B", "Growth")
        col4.metric("Approval Rating", f"{int(curr_app)}%", f"Turn {eco.turn}/20")

        # 2. Advisor Message
        # ------------------
        st.info(f"**Advisor:** {pol.get_advisor_comment(eco.state)}")

        # 3. Game Over Check
        # ------------------
        is_over, message = pol.check_game_over(eco.turn)
        if is_over:
            st.error(f"GAME OVER: {message}")
            if st.button("View Final Report"):
                pass # Just stops interaction
            return

        # 4. Charts Area
        # --------------
        tab1, tab2, tab3 = st.tabs(["📊 Main Indicators", "🌀 Phillips Curve", "🏛️ Fiscal & Debt"])
        
        with tab1:
            # Dual Axis Chart: Inflation vs Unemployment
            base = alt.Chart(df).encode(x='Turn')
            
            line_inf = base.mark_line(color='red').encode(
                y=alt.Y('Inflation', title='Inflation (%)'),
                tooltip=['Turn', 'Inflation']
            )
            
            line_unemp = base.mark_line(color='blue').encode(
                y=alt.Y('Unemployment', title='Unemployment (%)'),
                tooltip=['Turn', 'Unemployment']
            )
            
            st.altair_chart((line_inf + line_unemp).interactive(), use_container_width=True)
            
            st.markdown("**Blue:** Unemployment | **Red:** Inflation")

        with tab2:
            # Scatter Plot: Phillips Curve
            chart_pc = alt.Chart(df).mark_circle(size=60).encode(
                x=alt.X('Unemployment', scale=alt.Scale(domain=[0, 15]), title='Unemployment (%)'),
                y=alt.Y('Inflation', scale=alt.Scale(domain=[-2, 15]), title='Inflation (%)'),
                color=alt.value('purple'),
                tooltip=['Turn', 'Inflation', 'Unemployment']
            ).properties(title="The Phillips Curve Path")
            
            # Connect the dots to show path
            line_pc = alt.Chart(df).mark_line(color='gray', strokeDash=[5,5]).encode(
                x='Unemployment',
                y='Inflation',
                order='Turn'
            )
            
            st.altair_chart(chart_pc + line_pc, use_container_width=True)

        with tab3:
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("##### GDP Gap")
                # Bar chart of Output Gap
                df['Output_Gap'] = (df['GDP_Real'] - df['Potential_GDP'])
                chart_gap = alt.Chart(df).mark_bar().encode(
                    x='Turn',
                    y='Output_Gap',
                    color=alt.condition(
                        alt.datum.Output_Gap > 0,
                        alt.value("green"),
                        alt.value("orange")
                    )
                )
                st.altair_chart(chart_gap, use_container_width=True)
            
            with col_b:
                st.markdown("##### National Debt")
                chart_debt = alt.Chart(df).mark_area(color='darkred', opacity=0.5).encode(
                    x='Turn',
                    y='Debt'
                )
                st.altair_chart(chart_debt, use_container_width=True)


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    st.set_page_config(page_title="MacroSim 2025", layout="wide")
    
    # Initialize UI Controller
    app = SimulationUI()
    
    # Render
    app.render_sidebar()
    app.render_dashboard()