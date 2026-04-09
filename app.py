import streamlit as st
import pandas as pd
import numpy as np

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Advanced HMM Regime Model",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CUSTOM CSS (Institutional, Clean, Responsive)
# ==========================================
st.markdown("""
    <style>
    /* Typography and Spacing */    
    .main { max-width: 1100px; margin: 0 auto; font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
    
    /* Typography Upgrades */
    h1 { text-align: center; font-size: 2.8rem !important; font-weight: 900 !important; margin-bottom: 0.5rem !important; line-height: 1.25 !important; color: var(--text-color) !important; opacity: 0.95 !important; letter-spacing: -0.02em !important; }
    .subtitle { text-align: center; font-size: 1.3rem !important; color: var(--text-color) !important; opacity: 0.75 !important; font-style: italic !important; margin-bottom: 1.8rem !important; font-weight: 400 !important; }
    h2 { font-weight: 800 !important; font-size: 2.2rem !important; margin-top: 2.8rem !important; border-bottom: 1px solid rgba(128, 128, 128, 0.2) !important; padding-bottom: 0.5rem !important; margin-bottom: 1.5rem !important; color: var(--text-color) !important; opacity: 0.95 !important; letter-spacing: -0.01em !important; }
    h3 { font-weight: 700 !important; font-size: 1.6rem !important; margin-top: 2rem !important; margin-bottom: 1rem !important; color: var(--text-color) !important; opacity: 0.95 !important; }
    
    /* Paragraphs and Lists */
    div[data-testid="stMarkdownContainer"] p, div[data-testid="stMarkdownContainer"] li { font-size: 1.25rem !important; line-height: 1.6 !important; font-weight: 400 !important; color: var(--text-color) !important; opacity: 0.9 !important; } 
    div[data-testid="stMarkdownContainer"] li { margin-bottom: 0.5rem !important; }
    div[data-testid="stMarkdownContainer"] strong { font-weight: 700 !important; color: var(--text-color) !important; opacity: 1.0 !important; }
    
    /* Elegant blockquotes */
    .callout { 
        border-left: 5px solid #3B82F6 !important; 
        padding-left: 1.25rem !important; 
        margin: 1.4rem 0 !important; 
        color: var(--text-color) !important; 
        font-size: 1.25rem !important; 
        background-color: transparent !important;
    }

    
    /* UI Components */
    .section-badge { background-color: #2563EB; color: #FFFFFF !important; padding: 8px 16px; border-radius: 8px; font-size: 1.65rem; font-weight: 800; letter-spacing: 0.06em; text-transform: uppercase; display: inline-block; vertical-align: middle; margin-right: 12px; }
    .h2-text { position: relative; top: 3px; }
    .hero-box { background-color: var(--secondary-background-color); border: 1px solid rgba(128, 128, 128, 0.2); border-radius: 6px; padding: 20px 25px; margin: 1.5rem 0 2.5rem 0; }
    .hero-box-title { font-size: 1.2rem; font-weight: 700; color: #3B82F6; margin-bottom: 10px; text-transform: uppercase; letter-spacing: 0.05em; }
    .insight-box { border-left: 5px solid #8B5CF6 !important; padding: 1.25rem !important; margin: 1.5rem 0 !important; background-color: var(--secondary-background-color) !important; color: var(--text-color) !important; border-radius: 0 6px 6px 0 !important; }
    .insight-title { color: #8B5CF6; font-size: 1.25rem; font-weight: 800; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.05em; }

    .audit-box { border-left: 5px solid #F59E0B !important; padding: 1.25rem !important; margin: 1.5rem 0 !important; background-color: var(--secondary-background-color) !important; border-radius: 0 6px 6px 0 !important; }
    .findings-box { border-left: 5px solid #10B981 !important; padding: 1.2rem !important; margin: 2rem 0 !important; background-color: var(--secondary-background-color) !important; border-radius: 0 6px 6px 0 !important; }
    .findings-title { font-size: 1.25rem; font-weight: 700; margin-bottom: 10px; color: #10B981; text-transform: uppercase; letter-spacing: 0.05em; }

    
    /* Stats & Tables */
    .stat-row { display: flex; gap: 20px; margin-bottom: 25px; margin-top: 15px; }
    .stat-box { flex: 1; background-color: var(--secondary-background-color) !important; border: 1px solid rgba(128, 128, 128, 0.2) !important; border-radius: 6px !important; padding: 15px !important; text-align: center !important; margin-bottom: 15px !important; }
    .stat-value { font-size: 2.0rem !important; font-weight: 700 !important; color: var(--text-color) !important; margin-bottom: 2px !important; line-height: 1 !important;}
    .stat-label { font-size: 0.85rem !important; font-weight: 600 !important; color: var(--text-color) !important; opacity: 0.6 !important; text-transform: uppercase !important; letter-spacing: 0.05em !important;}
    
    /* Sidebar Table of Contents */
    .toc-link { text-decoration: none !important; font-size: 1.2rem !important; text-align: center !important; display: block !important; padding: 12px 10px !important; font-weight: 500 !important; color: var(--text-color) !important; opacity: 0.8 !important; transition: opacity 0.2s ease-in-out !important; }
    .toc-link:hover { opacity: 1.0 !important; color: #FBBF24 !important; background-color: rgba(251, 191, 36, 0.05) !important; text-decoration: none !important; }
    
    /* Sidebar Author Card */
    .author-card-fixed { background-color: #0F172A !important; border-top: 2px solid #FBBF24 !important; border-radius: 10px !important; padding: 12px 10px !important; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4) !important; margin-bottom: 15px !important; text-align: center !important; }
    .sidebar-label-fixed { color: #FBBF24 !important; font-weight: 900 !important; letter-spacing: 0.15em !important; text-transform: uppercase !important; font-size: 1.03rem !important; display: block !important; margin-bottom: 2px !important; }
    .sidebar-author-fixed { color: #FFFFFF !important; font-weight: 800 !important; font-size: 1.34rem !important; display: block !important; line-height: 1.1 !important; }
    .sidebar-sub-fixed { color: rgba(255, 255, 255, 0.65) !important; font-style: italic !important; font-weight: 400 !important; font-size: 1.05rem !important; display: block !important; margin-top: 8px !important; border-top: 1px solid rgba(255, 255, 255, 0.1); padding-top: 6px !important; }
    
    /* Mobile Overrides */
    .mobile-sidebar-hint { display: none; }
    .mobile-menu-badge { display: none; } 
    
    @media (max-width: 768px) {
        .mobile-sidebar-hint { display: block !important; background-color: rgba(59, 130, 246, 0.1); color: #3B82F6; padding: 12px; border-radius: 8px; text-align: center; font-weight: 500; font-size: 0.95rem; margin-top: 25px; border: 1px solid rgba(59, 130, 246, 0.3); line-height: 1.4; }
        .mobile-menu-badge { display: block !important; position: fixed; top: 14px; left: 55px; background-color: #2563EB; color: white; padding: 4px 10px; border-radius: 12px; font-size: 0.85rem; font-weight: 800; letter-spacing: 0.05em; text-transform: uppercase; z-index: 999999 !important; box-shadow: 0 2px 5px rgba(0,0,0,0.2); animation: pulse 2s infinite; pointer-events: none; transition: opacity 0.2s ease-in-out, visibility 0.2s; }
        @keyframes pulse { 0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(37, 99, 235, 0.7); } 70% { transform: scale(1); box-shadow: 0 0 0 6px rgba(37, 99, 235, 0); } 100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(37, 99, 235, 0); } }
        .stApp:has([data-testid="stSidebar"][aria-expanded="true"]) .mobile-menu-badge { opacity: 0 !important; visibility: hidden !important; }
    }
    
    /* VS Comparison Box for Architecture */
    .vs-container { display: flex; gap: 25px; margin: 25px 0; }
    .vs-col { flex: 1; padding: 25px; border-radius: 10px; border: 1px solid rgba(128,128,128,0.2); background-color: var(--secondary-background-color); box-shadow: 0 4px 6px rgba(0,0,0,0.02); }
    .vs-col.bad { border-top: 5px solid #EF4444; }
    .vs-col.good { border-top: 5px solid #10B981; }
    .vs-title { font-weight: 800; font-size: 1.2rem; margin-bottom: 20px; text-transform: uppercase; letter-spacing: 0.05em;}
    .vs-title.bad-title { color: #EF4444; }
    .vs-title.good-title { color: #10B981; }
    .vs-item { margin-bottom: 18px; font-size: 1.05rem; line-height: 1.5; }
    .vs-item strong { display: block; color: var(--text-color); font-size: 1.1rem; margin-bottom: 4px; }
    .vs-item span { opacity: 0.85; display: block; }
    

    /* Style for Streamlit Tabs */
    button[data-baseweb="tab"] { font-size: 1.15rem !important; font-weight: 600 !important; padding-top: 1rem !important; padding-bottom: 1rem !important; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# DATA GENERATORS (Strictly from results_HMM.txt)
# ==========================================
@st.cache_data
def get_macro_profile():
    return pd.DataFrame([
        {'Regime': 'Calm', 'Alloc': '53.9%', 'Ann Ret': '-78.70%', 'Ann Vol': '5.06%', 'Sharpe': '-15.55', 'Kurtosis': '2.6'},
        {'Regime': 'Turbulent', 'Alloc': '28.0%', 'Ann Ret': '43.18%', 'Ann Vol': '2.24%', 'Sharpe': '19.27', 'Kurtosis': '1.3'},
        {'Regime': 'Crisis', 'Alloc': '18.1%', 'Ann Ret': '162.78%', 'Ann Vol': '5.81%', 'Sharpe': '28.01', 'Kurtosis': '7.5'}
    ])

@st.cache_data
def get_meso_profile():
    return pd.DataFrame([
        {'Regime': 'Calm', 'Alloc': '64.6%', 'Ann Ret': '76.79%', 'Ann Vol': '6.41%', 'Sharpe': '11.98', 'Kurtosis': '12.9'},
        {'Regime': 'Turbulent', 'Alloc': '22.6%', 'Ann Ret': '-168.03%', 'Ann Vol': '7.45%', 'Sharpe': '-22.56', 'Kurtosis': '7.6'},
        {'Regime': 'Crisis', 'Alloc': '12.8%', 'Ann Ret': '-96.22%', 'Ann Vol': '9.31%', 'Sharpe': '-10.34', 'Kurtosis': '1.9'}
    ])

@st.cache_data
def get_micro_profile():
    return pd.DataFrame([
        {'Regime': 'Calm', 'Alloc': '81.9%', 'Ann Ret': '0.38%', 'Ann Vol': '4.09%', 'Sharpe': '0.09', 'Kurtosis': '0.8'},
        {'Regime': 'Turbulent', 'Alloc': '9.9%', 'Ann Ret': '-473.21%', 'Ann Vol': '13.48%', 'Sharpe': '-35.11', 'Kurtosis': '4.0'},
        {'Regime': 'Crisis', 'Alloc': '8.1%', 'Ann Ret': '564.32%', 'Ann Vol': '15.68%', 'Sharpe': '35.99', 'Kurtosis': '2.8'}
    ])

@st.cache_data
def get_hierarchical_matrices():
    m1d = pd.DataFrame({'To Calm': ['0.9450', '0.0324', '0.0177'], 'To Turb': ['0.0235', '0.9358', '0.0177'], 'To Crisis': ['0.0315', '0.0319', '0.9645']}, index=['From Calm', 'From Turb', 'From Crisis'])
    m4h = pd.DataFrame({'To Calm': ['0.9647', '0.0786', '0.0842'], 'To Turb': ['0.0177', '0.8760', '0.0368'], 'To Crisis': ['0.0177', '0.0454', '0.8790']}, index=['From Calm', 'From Turb', 'From Crisis'])
    m1h = pd.DataFrame({'To Calm': ['0.9640', '0.1645', '0.1645'], 'To Turb': ['0.0178', '0.7868', '0.0487'], 'To Crisis': ['0.0178', '0.0487', '0.7868']}, index=['From Calm', 'From Turb', 'From Crisis'])
    return m1d, m4h, m1h

@st.cache_data
def get_calibration_data():
    return pd.DataFrame([
        {'Metric': 'Global Brier Score', 'Value': '0.0702'},
        {'Metric': 'Global Log Loss', 'Value': '0.2281'},
        {'Metric': 'Calm Brier / Log Loss', 'Value': '0.0305 / 0.1250'},
        {'Metric': 'Turbulent Brier / Log Loss', 'Value': '0.3042 / 0.8209'},
        {'Metric': 'Crisis Brier / Log Loss', 'Value': '0.1833 / 0.5419'}
    ])

@st.cache_data
def get_mrm_audit():
    return pd.DataFrame({
        "MRM Metric": ["TRUE RCM Clarity", "Extreme Volatility Brier Score", "Residual ACF (1H)", "VaR 99 Breach Rate"],
        "Target Standard": ["> 75.0%", "< 0.1000", "~ 0.000", "~ 1.00%"],
        "Empirical Result": ["51.60% (FAIL)", "0.0702 (PASS)", "0.0065 (PASS)", "0.55% (PASS)"]
    })

@st.cache_data
def get_vol_benchmark():
    return pd.DataFrame({
        "Metric": ["RMSE", "Error Mean", "Error Std", "Spearman IC (Raw)", "Spearman IC (Smoothed)"],
        "MS-GARCH (HMM Engine)": ["0.037043", "0.001386", "0.037017", "+0.3653", "+0.5338"],
        "Causal GARCH (Baseline)": ["0.037410", "0.002304", "0.037339", "+0.3462", "+0.5264"]
    })

@st.cache_data
def get_regime_diagnostics():
    return pd.DataFrame([
        {"Diagnostic Test": "1. Volatility Stratification", "Metric Evaluated": "Monotonic OOS Annualized Volatility", "Result": "PASS (6.04% < 8.81% < 9.74%)"},
        {"Diagnostic Test": "2. Tail-Risk Isolation", "Metric Evaluated": "CVaR-99 Monotonicity", "Result": "PASS (-34.6bps < Uncond < -48.8bps)"},
        {"Diagnostic Test": "3. Empirical OOS Churn", "Metric Evaluated": "Empirical Avg Dwell Time", "Result": "PASS (4.54 Hours)"},
        {"Diagnostic Test": "4. Distributional Purity", "Metric Evaluated": "Kolmogorov-Smirnov (KS) Test", "Result": "PASS (p=1.26e-154, distinct states)"},
        {"Diagnostic Test": "5. Early Warning Power", "Metric Evaluated": "Fwd Drawdown in T+12H", "Result": "FAIL (-0.18%, lagging indicator)"},
        {"Diagnostic Test": "6. Heteroskedasticity Absorption", "Metric Evaluated": "Conditional ACF < Unconditional", "Result": "FAIL (Calm ACF > Uncond ACF)"},
        {"Diagnostic Test": "7. Empirical Transition Stability", "Metric Evaluated": "OOS Persistence matches IS Bounds", "Result": "PASS (Calm->Calm 0.846)"}
    ])

@st.cache_data
def get_empirical_transition_matrix():
    return pd.DataFrame({
        'From State': ['Calm', 'Turbulent', 'Crisis'],
        'To Calm': ['0.846', '0.199', '0.156'],
        'To Turbulent': ['0.065', '0.721', '0.162'],
        'To Crisis': ['0.089', '0.080', '0.682']
    })

@st.cache_data
def get_tensor_3d_data():
    return pd.DataFrame({
        'Macro 1D / Meso 4H': [
            'Calm / Calm', 'Calm / Turb', 'Calm / Crisis', 
            'Turb / Calm', 'Turb / Turb', 'Turb / Crisis', 
            'Crisis / Calm', 'Crisis / Turb', 'Crisis / Crisis'
        ],
        'Micro: Calm': ['+18.20%', '-27.41%', '-1.59%', '-4.55%', '-13.17%', '-0.84%', '+15.52%', '-3.76%', '+0.60%'],
        'Micro: Turb': ['+14.56%', '-22.84%', '+12.09%', '+1.32%', '-1.49%', '+0.87%', '+0.75%', '-0.13%', '+0.92%'],
        'Micro: Crisis': ['-2.93%', '-1.02%', '+3.88%', '+1.17%', '-0.01%', '-3.21%', '-15.41%', '+0.05%', '-1.84%']
    })

@st.cache_data
def get_feature_ablation():
    return pd.DataFrame([
        {'Feature': 'Micro_Vol_Spike', 'Avg Weight': '-0.001183'},
        {'Feature': 'Hazard_Delta', 'Avg Weight': '+0.000730'},
        {'Feature': 'Sig_MR', 'Avg Weight': '+0.000642'},
        {'Feature': 'Price_Z_4h', 'Avg Weight': '-0.000642'},
        {'Feature': 'Tail_Asymmetry', 'Avg Weight': '-0.000447'},
        {'Feature': 'Entropy_Slope', 'Avg Weight': '+0.000406'},
        {'Feature': 'MR_x_Calm', 'Avg Weight': '+0.000394'},
        {'Feature': 'Micro_Dwell', 'Avg Weight': '-0.000296'},
        {'Feature': 'Micro_Entropy', 'Avg Weight': '+0.000277'},
        {'Feature': 'TSMOM_1h', 'Avg Weight': '-0.000238'}
    ])

@st.cache_data
def get_top_10_states():
    return pd.DataFrame([
        {'State Phase': 'Bear Trend (Macro/Meso/Micro aligned)', 'P&L (%)': '+8.54%'},
        {'State Phase': 'Bear Setup (Macro:Bull, Meso:Bear, Micro:Bear)', 'P&L (%)': '+4.51%'},
        {'State Phase': 'Crisis Pullback (Macro:Bear, Meso:Crisis, Micro:Turbulent)', 'P&L (%)': '+4.18%'},
        {'State Phase': 'Micro Bear inside Bull Macro', 'P&L (%)': '+3.23%'},
        {'State Phase': 'Turbulent Chop (Macro:Bear, Meso:Turb, Micro:Turb)', 'P&L (%)': '+3.00%'},
        {'State Phase': 'Macro Bear Relief Rally', 'P&L (%)': '+2.80%'}
    ])

@st.cache_data
def get_tear_sheet():
    return pd.DataFrame({
        'Metric': ['Net Ann. Return', 'Ann. Volatility', 'Net Sharpe', 'Sortino', 'Max Drawdown', 'Win Rate', 'Profit Factor'],
        'Vanilla ML (No Regime)': ['-1.43%', '13.51%', '-0.11', '-0.15', '-31.57%', '48.25%', '1.00'],
        'Regime ML (Optimal)': ['-1.59%', '6.59%', '-0.24', '-0.35', '-16.51%', '43.98%', '0.99']
    })

@st.cache_data
def get_execution_overlays():
    return pd.DataFrame([
        {"Execution Overlay": "1. Base ML Signal", "Sharpe": "-0.24", "Ann Ret": "-1.59%", "MDD": "-16.51%"},
        {"Execution Overlay": "2. Sparse Threshold Signal", "Sharpe": "+0.05", "Ann Ret": "+0.22%", "MDD": "-10.11%"},
        {"Execution Overlay": "3. Hard Regime Routed", "Sharpe": "-0.85", "Ann Ret": "-9.14%", "MDD": "-33.64%"},
        {"Execution Overlay": "4. Hazard Delta Exit", "Sharpe": "-0.29", "Ann Ret": "-1.85%", "MDD": "-17.53%"},
        {"Execution Overlay": "5. High Conviction Filter", "Sharpe": "-0.29", "Ann Ret": "-1.99%", "MDD": "-17.73%"},
        {"Execution Overlay": "6. Signal & Regime Agreement", "Sharpe": "-0.36", "Ann Ret": "-2.03%", "MDD": "-16.54%"},
        {"Execution Overlay": "7. Pure Conviction Sizing", "Sharpe": "-0.29", "Ann Ret": "-3.42%", "MDD": "-27.61%"},
    ])

@st.cache_data
def get_ic_heatmap():
    return pd.DataFrame([
        {'Horizon': 'T+1 Hour', 'Global IC': '+0.0281', 'Calm IC': '+0.0369', 'Turbulent IC': '+0.0285', 'Crisis IC': '+0.0099'},
        {'Horizon': 'T+4 Hours', 'Global IC': '+0.0157', 'Calm IC': '+0.0197', 'Turbulent IC': '+0.0219', 'Crisis IC': '+0.0009'},
        {'Horizon': 'T+24 Hours', 'Global IC': '+0.0138', 'Calm IC': '+0.0045', 'Turbulent IC': '+0.0299', 'Crisis IC': '+0.0119'},
    ])

# @st.cache_data
# def get_slippage_data():
#     return pd.DataFrame({
#         'Added Spread': ['+0.0 bps (Base TCA)', '+0.5 bps', '+1.0 bps', '+2.0 bps'],
#         'Net Ann. Return': ['-3.91%', '-25.91%', '-47.92%', '-91.93%'],
#         'Net Sharpe': ['-0.53', '-3.51', '-6.45', '-12.17'],
#         'Max Drawdown': ['-28.74%', '-74.44%', '-91.67%', '-99.14%']
#     })

@st.cache_data
def get_strat_sharpe_matrix():
    return pd.DataFrame([
        {'Regime': 'Calm', 'Mean Reversion': '1.69', 'Momentum': '-1.30', 'Vanilla ML': '1.40', 'Regime ML': '0.08'},
        {'Regime': 'Turbulent', 'Mean Reversion': '1.23', 'Momentum': '0.71', 'Vanilla ML': '1.51', 'Regime ML': '-0.13'},
        {'Regime': 'Crisis', 'Mean Reversion': '1.45', 'Momentum': '-0.33', 'Vanilla ML': '-4.09', 'Regime ML': '-1.27'}
    ])
    
@st.cache_data
def get_meta_regime_data():
    return pd.DataFrame([
        {'Meta-Regime': 'Micro Shock', 'Time Active': '21.95%', 'Net P&L (%)': '+17.15%', 'Directional IC': '+0.0332'},
        {'Meta-Regime': 'Macro Overhang', 'Time Active': '17.86%', 'Net P&L (%)': '+4.98%', 'Directional IC': '+0.0481'},
        {'Meta-Regime': 'Absolute Calm', 'Time Active': '19.93%', 'Net P&L (%)': '-1.25%', 'Directional IC': '+0.0382'},
        {'Meta-Regime': 'Mixed/Transitional', 'Time Active': '17.59%', 'Net P&L (%)': '-8.74%', 'Directional IC': '+0.0349'},
        {'Meta-Regime': 'Spillover Contagion', 'Time Active': '14.09%', 'Net P&L (%)': '-14.21%', 'Directional IC': '+0.0182'},
        {'Meta-Regime': 'Total Crisis', 'Time Active': '8.58%', 'Net P&L (%)': '-4.50%', 'Directional IC': '-0.0359'}
    ])
    
@st.cache_data
def get_narrative_scenarios():
    return pd.DataFrame([
        {'Alpha Scenario': 'ML Baseline (No Filter)', 'Theoretical P&L (%)': '+43.72%'},
        {'Alpha Scenario': 'Thin-Liquidity Reversion', 'Theoretical P&L (%)': '+1.37%'},
        {'Alpha Scenario': 'Pre-Spillover Compression', 'Theoretical P&L (%)': '+0.11%'},
        {'Alpha Scenario': 'Post-Spillover Exhaustion', 'Theoretical P&L (%)': '-2.17%'},
        {'Alpha Scenario': 'Directed Spillover', 'Theoretical P&L (%)': '-12.08%'}
    ])
    
@st.cache_data
def get_dm_test_metrics():
    return pd.DataFrame([
        {'Metric': 'RMSE', 'MS-GARCH': '0.037043', 'Causal GARCH': '0.037410', 'Delta': '-0.000367'},
        {'Metric': 'Spearman IC (Raw)', 'MS-GARCH': '0.3653', 'Causal GARCH': '0.3462', 'Delta': '+0.0191'},
        {'Metric': 'Spearman IC (Smoothed)', 'MS-GARCH': '0.5338', 'Causal GARCH': '0.5264', 'Delta': '+0.0074'}
    ])

# ==========================================
# SIDEBAR NAVIGATION
# ==========================================
with st.sidebar:
    st.markdown("""
    <div class="author-card-fixed">
        <span class="sidebar-label-fixed">AUTHOR</span>
        <span class="sidebar-author-fixed">Jayesh Chaudhary</span>
        <span class="sidebar-sub-fixed">Quantitative Researcher</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.title("Research Outline")
    st.markdown("""
    <a href="#0-setup" target="_self" class="toc-link">0. Problem Setup & Approach</a>
    <a href="#1-market-dynamics" target="_self" class="toc-link">1. Market Dynamics & Calibration</a>
    <a href="#2-mrm" target="_self" class="toc-link">2. Model Risk & Baseline Benchmarking</a>
    <a href="#3-diagnostics" target="_self" class="toc-link">3. Regime Diagnostics</a>
    <a href="#4-allocation" target="_self" class="toc-link">4. Multi-Timeframe Integration</a>
    <a href="#5-meta-regimes" target="_self" class="toc-link">5. Meta-Regime Attribution</a>
    <a href="#6-theoretical-limitations" target="_self" class="toc-link">6. Structural Limitations</a>
    <a href="#7-conclusion" target="_self" class="toc-link">7. Final Verdict</a>
    
    <div class="mobile-sidebar-hint">
        <strong>Tap outside</strong> or <strong>swipe left</strong> to close menu.
    </div>
    """, unsafe_allow_html=True)

    

# ==========================================
# MAIN DOCUMENT
# ==========================================
st.markdown('<div class="mobile-menu-badge">👈 Topics Menu</div>', unsafe_allow_html=True)

st.markdown("""
    <div style="text-align: center; margin-top: 1rem; margin-bottom: 0.2rem;">
        <span style="background-color: #F59E0B; color: #111827; padding: 4px 14px; border-radius: 20px; font-size: 0.85rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.05em; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">Research in Progress (Beta)</span>
    </div>
""", unsafe_allow_html=True)

st.markdown("<h1>Advanced Triple Regime Markov Switching GARCH Model</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>An empirical out-of-sample analysis of a custom HMM regime model on EUR/USD.</p>", unsafe_allow_html=True)

st.markdown("""
    <div style='text-align: center; margin-top: -1.0rem; margin-bottom: 3rem;'>
        <span style='font-size: 1.0rem; color: var(--text-color); opacity: 0.5; font-weight: 400; letter-spacing: 0.15em; text-transform: uppercase;'>
            Research by <strong>Jayesh Chaudhary</strong>
        </span>
    </div>
""", unsafe_allow_html=True)

# ------------------------------------------
# QUICK STATS BAR
# ------------------------------------------
col_a, col_b, col_c = st.columns(3)
box_css = "background-color: rgba(16, 185, 129, 0.03); border: 1px solid rgba(16, 185, 129, 0.2); border-radius: 8px; padding: 20px 15px; text-align: center; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.02);"
val_css = "font-size: 1.7rem; font-weight: 800; color: #10B981; margin-bottom: 6px; line-height: 1;"
lbl_css = "font-size: 0.90rem; font-weight: 700; color: var(--text-color); opacity: 0.75; text-transform: uppercase; letter-spacing: 0.05em;"

with col_a:
    st.markdown(f"<div style='{box_css}'><div style='{lbl_css}'>Target Asset</div><div style='{val_css}'>EUR/USD</div></div>", unsafe_allow_html=True)
with col_b:
    st.markdown(f"<div style='{box_css}'><div style='{lbl_css}'>Duration</div><div style='{val_css}'>2015 - 2025 (10 Yrs)</div></div>", unsafe_allow_html=True)
with col_c:
    st.markdown(f"<div style='{box_css}'><div style='{lbl_css}'>Regime Hierarchy</div><div style='{val_css}'>1H, 4H, 1D</div></div>", unsafe_allow_html=True)


# ------------------------------------------
# KEY CONTRIBUTIONS HERO BOX
# ------------------------------------------
# ------------------------------------------
# KEY CONTRIBUTIONS HERO BOX
# ------------------------------------------
st.markdown("""
<div class="hero-box" style="background-color: var(--secondary-background-color); border: 1px solid rgba(128, 128, 128, 0.2); border-radius: 8px; padding: 25px 30px; margin: 1.5rem 0 2.5rem 0; box-shadow: 0 4px 6px rgba(0,0,0,0.02);">
    <div style="font-size: 1.25rem; font-weight: 800; color: #2563EB; margin-bottom: 15px; text-transform: uppercase; letter-spacing: 0.05em;">Key Contributions</div>
    <ul style="margin-bottom: 0;">
        <li><strong>Multi-Timeframe Detection:</strong> We track regimes across daily, 4-hour, and 1-hour charts simultaneously using a nested MS-GARCH model. This stops the model from overreacting to short-term intraday noise.</li>
        <li><strong>Dynamic Transitions (TVTP):</strong> Instead of assuming the probability of a market shift is constant, we use real-time liquidity drivers to update transition probabilities continuously.</li>
        <li><strong>Fixing the 'Flickering' Problem:</strong> Standard models change their minds too fast. We wrote a custom C-compiled filter with Bayesian stickiness priors that mathematically forces the model to commit to a regime, cutting down on false signals.</li>
        <li><strong>Handling Fat Tails:</strong> We swapped standard Gaussian distributions for Skew-t distributions. This keeps the math stable during extreme market crashes rather than breaking the optimizer.</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# ==========================================
# SECTION 0: PROBLEM SETUP & APPROACH
# ==========================================
st.markdown("<h2 id='0-setup'><span class='section-badge'>Phase 0</span><span class='h2-text'> Problem Setup & Approach</span></h2>", unsafe_allow_html=True)

st.markdown("""
Financial time-series data isn't stationary. The way volatility trends and mean-reverts changes completely during a liquidity crisis compared to a calm market. This project builds a custom econometric engine to map these regime shifts in real-time.
""")

st.markdown("""
<div class="vs-container">
    <div class="vs-col bad">
        <div class="vs-title bad-title">Standard HMM (The Flaws)</div>
        <div class="vs-item">
            <strong>1. The "Flickering" Problem</strong>
            <span>Standard models react instantly to noise, rapidly flipping back and forth between "Calm" and "Crisis." This lack of commitment destroys P&L via relentless transaction fees.</span>
        </div>
        <div class="vs-item">
            <strong>2. Gaussian Naivety</strong>
            <span>They assume returns follow a normal bell curve. When a massive black swan event occurs, the math simply breaks.</span>
        </div>
        <div class="vs-item">
            <strong>3. Path-Dependence Explosion</strong>
            <span>To avoid the math exploding to infinity, standard models assume variance is constant within a state. They completely ignore Volatility Clustering.</span>
        </div>
        <div class="vs-item">
            <strong>4. Static Transitions</strong>
            <span>They assume the probability of entering a crisis tomorrow is exactly the same as yesterday, ignoring real-time macro shocks.</span>
        </div>
    </div>
    <div class="vs-col good">
        <div class="vs-title good-title">This project (The Fixes)</div>
        <div class="vs-item">
            <strong>1. Dual-Lock Hamilton Filter</strong>
            <span>We built a filter with heavy diagonal Bayesian priors (<code>stickiness=250.0</code>). It acts as a mathematical shock-absorber, forcing the model to stay in a regime until there's overwhelming evidence to switch.</span>
        </div>
        <div class="vs-item">
            <strong>2. Hansen's Skew-t Emissions</strong>
            <span>We natively model asymmetric left-tail crashes and fat-tailed black swans, ensuring the optimizer actually survives extreme outliers.</span>
        </div>
        <div class="vs-item">
            <strong>3. AR(1) MS-GARCH</strong>
            <span>Following Haas et al. (2004), we solve path-dependence by running parallel, independent GARCH(1,1) tracks <i>within</i> each regime.</span>
        </div>
        <div class="vs-item">
            <strong>4. Logistic TVTP</strong>
            <span>Time-Varying Transition Probabilities map exogenous macro shocks to state transitions, allowing the model to adapt continuously in real-time.</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="audit-box">
<strong>Data Integrity & The "Look-Ahead Bias" Wall</strong><br>
Before running any models, we have to guarantee no future data leaks into the past. A common mistake is calculating Z-scores across the whole dataset, which accidentally bleeds future volatility spikes (like the 2020 pandemic) into 2018 training data. <br><br>
We built a strict cutoff wall. Statistical parameters are calculated <em>only</em> on the training data. Furthermore, because we predict multi-hour targets, we drop the final rows of the training set (<code>[:-24]</code>) to create a "dead zone" between the train and test splits, preventing any future prices from bleeding backward.<br><br>
<strong>Preventing Rolling Window Data Loss (Zero-Amnesia WFA):</strong><br>
In standard Walk-Forward Analysis, rolling features (like a 24-hour moving average) break and return `NaN` at the start of every new test set. We engineered a "Zero-Amnesia" pipeline that dynamically fuses the tail of the Train set to the head of the Test set. This ensures all path-dependent technical indicators roll seamlessly across quarter-boundaries without ever peeking into the future.
</div>
""", unsafe_allow_html=True)



# ==========================================
# SECTION 1: MARKET DYNAMICS
# ==========================================
st.markdown("""
<div style="background-color: rgba(16, 185, 129, 0.1); border-left: 4px solid #10B981; padding: 15px; margin-bottom: 20px; border-radius: 4px;">
<strong>Crucial Note on Methodology:</strong> Everything in this section—the return profiles, the volatilities, and the transition matrices—is calculated <strong>strictly Out-Of-Sample (OOS)</strong>. This is not an in-sample curve fit. This is how the asset <em>actually behaved</em> when the model was predicting blind on unseen data over 4 years.
</div>
""", unsafe_allow_html=True)

st.markdown("Before testing any trading overlays, we first need to understand how EUR/USD naturally behaves in different states. We mapped the underlying return distributions across three hierarchical timeframes (Daily, 4-Hour, and Hourly) to see exactly what 'Calm' vs 'Crisis' looks like in the data.")

tab_m1, tab_m2, tab_m3 = st.tabs(["Macro (1-Day)", "Meso (4-Hour)", "Micro (1-Hour)"])
m1d, m4h, m1h = get_hierarchical_matrices()

with tab_m1:
    with st.container(border=True):
        col_t1, col_t2 = st.columns([1, 1])
        with col_t1:
            st.markdown("**Out-of-Sample (OOS) Return & Volatility Profile**")
            st.dataframe(get_macro_profile(), use_container_width=True, hide_index=True)
        with col_t2:
            st.markdown("**Out-of-Sample (OOS) Transition Matrix**")
            st.dataframe(m1d, use_container_width=True)
        
        st.markdown("""
        <div style="background-color: rgba(128,128,128,0.03); border: 1px solid rgba(128,128,128,0.1); border-radius: 8px; padding: 15px; margin-top: 10px;">
            <div style="text-align: center; font-size: 0.95rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; opacity: 0.7; margin-bottom: 10px;">Calculated Average Time in Regime</div>
            <div class="stat-row" style="margin-top: 0px; margin-bottom: 0px;">
            <div class="stat-box" style="box-shadow: none; border: none !important; background: transparent !important;"><div class="stat-value">18.18 Days</div><div class="stat-label">Calm State</div></div>
            <div class="stat-box" style="box-shadow: none; border: none !important; background: transparent !important;"><div class="stat-value">15.58 Days</div><div class="stat-label">Turbulent State</div></div>
            <div class="stat-box" style="box-shadow: none; border: none !important; background: transparent !important;"><div class="stat-value">28.17 Days</div><div class="stat-label">Crisis State</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

with tab_m2:
    with st.container(border=True):
        col_t1, col_t2 = st.columns([1, 1])
        with col_t1:
            st.markdown("**Out-of-Sample (OOS) Return & Volatility Profile**")
            st.dataframe(get_meso_profile(), use_container_width=True, hide_index=True)
        with col_t2:
            st.markdown("**Out-of-Sample (OOS) Transition Matrix**")
            st.dataframe(m4h, use_container_width=True)
            
        st.markdown("""
        <div style="background-color: rgba(128,128,128,0.03); border: 1px solid rgba(128,128,128,0.1); border-radius: 8px; padding: 15px; margin-top: 10px;">
            <div style="text-align: center; font-size: 0.95rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; opacity: 0.7; margin-bottom: 10px;">Calculated Average Time in Regime</div>
            <div class="stat-row" style="margin-top: 0px; margin-bottom: 0px;">
            <div class="stat-box" style="box-shadow: none; border: none !important; background: transparent !important;"><div class="stat-value">4.7 Days</div><div class="stat-label">Calm State</div></div>
            <div class="stat-box" style="box-shadow: none; border: none !important; background: transparent !important;"><div class="stat-value">1.3 Days</div><div class="stat-label">Turbulent State</div></div>
            <div class="stat-box" style="box-shadow: none; border: none !important; background: transparent !important;"><div class="stat-value">1.4 Days</div><div class="stat-label">Crisis State</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

with tab_m3:
    with st.container(border=True):
        col_t1, col_t2 = st.columns([1, 1])
        with col_t1:
            st.markdown("**Out-of-Sample (OOS) Return & Volatility Profile**")
            st.dataframe(get_micro_profile(), use_container_width=True, hide_index=True)
        with col_t2:
            st.markdown("**Out-of-Sample (OOS) Transition Matrix**")
            st.dataframe(m1h, use_container_width=True)
            
        st.markdown("""
        <div style="background-color: rgba(128,128,128,0.03); border: 1px solid rgba(128,128,128,0.1); border-radius: 8px; padding: 15px; margin-top: 10px;">
            <div style="text-align: center; font-size: 0.95rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; opacity: 0.7; margin-bottom: 10px;">Calculated Average Time in Regime</div>
            <div class="stat-row" style="margin-top: 0px; margin-bottom: 0px;">
            <div class="stat-box" style="box-shadow: none; border: none !important; background: transparent !important;"><div class="stat-value">28.17 Hrs</div><div class="stat-label">Calm State</div></div>
            <div class="stat-box" style="box-shadow: none; border: none !important; background: transparent !important;"><div class="stat-value">4.69 Hrs</div><div class="stat-label">Turbulent State</div></div>
            <div class="stat-box" style="box-shadow: none; border: none !important; background: transparent !important;"><div class="stat-value">4.69 Hrs</div><div class="stat-label">Crisis State</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("#### Tying Math to Market Reality (Bayesian Stickiness)")
st.markdown("""
We intentionally set varying Bayesian `stickiness_priors` during training (100.0 for Macro, 150.0 for Meso, and 200.0 for Micro) to force the model to respect actual institutional market cycles. When we calculate the expected average time spent in each regime ($1 / (1 - P)$), the results perfectly align with real-world trading mechanics:
* **Macro Timeframe (~15 to 28 Days):** Captures monthly fund rebalancing, overarching macro narratives (inflation/rates), and earnings cycles.
* **Meso Timeframe (~1 to 4 Days):** Captures the exact duration of volatility shocks surrounding Central Bank rate decisions, NFP prints, and geopolitical headlines.
* **Micro Timeframe (~4.69 Hours):** Captures intraday liquidity windows, such as the overlap between the London and New York trading sessions.
""")

st.markdown("#### The Reality of Intraday Market Noise")
st.markdown("By calculating the expected average time spent in each regime mathematically from our transition matrices ($1 / (1 - P)$), we see exactly how fast the Micro (1-Hour) timeframe moves. A 'Crisis' state only persists for about 4.69 hours on average before reverting. This mathematically proves that high-frequency FX shocks are extremely brief, creating a massive execution hurdle for trend-following strategies.")



st.image("transition_heatmaps.png", caption="Transition Matrix Heatmaps visually confirming state persistence", use_container_width=True)

st.markdown("#### Dynamic Matrix Drivers (TVTP Feature Engineering)")
st.markdown("""
Standard HMMs are "blind"—they use a static transition matrix that never changes. To make our model react to real-world liquidity and macro events, we engineered **Time-Varying Transition Probabilities (TVTP)**. 

We built custom exogenous drivers for each timeframe. This ensures the Daily model doesn't flip into a Crisis due to 1-hour noise, while the Hourly model remains highly sensitive to session overlap liquidity.
""")

col_tvtp1, col_tvtp2, col_tvtp3 = st.columns(3)
with col_tvtp1:
    st.markdown("""
    <div class="stat-box" style="padding: 15px; text-align: left !important; height: 100%;">
    <div class="stat-label" style="margin-bottom: 5px; color: #2563EB !important;">1H Micro (Liquidity)</div>
    <span style="font-size: 0.90rem; opacity: 0.85;">Driven purely by intraday flow.</span>
    <ul style="font-size: 0.90rem; margin-top: 10px; padding-left: 20px; opacity: 0.9;">
        <li><strong>70%</strong> Intraday Volatility (Meta Z-Score)</li>
        <li><strong>30%</strong> Hour-of-Day Seasonality</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
with col_tvtp2:
    st.markdown("""
    <div class="stat-box" style="padding: 15px; text-align: left !important; height: 100%;">
    <div class="stat-label" style="margin-bottom: 5px; color: #2563EB !important;">4H Meso (The Bridge)</div>
    <span style="font-size: 0.90rem; opacity: 0.85;">Blends intraday flow with macro state.</span>
    <ul style="font-size: 0.90rem; margin-top: 10px; padding-left: 20px; opacity: 0.9;">
        <li><strong>45%</strong> Intraday Volatility</li>
        <li><strong>35%</strong> Day-of-Week Seasonality</li>
        <li><strong>20%</strong> Macro Stress Index</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
with col_tvtp3:
    st.markdown("""
    <div class="stat-box" style="padding: 15px; text-align: left !important; height: 100%;">
    <div class="stat-label" style="margin-bottom: 5px; color: #2563EB !important;">1D Macro (The Anchor)</div>
    <span style="font-size: 0.90rem; opacity: 0.85;">Ignores micro noise. Tracks fundamentals.</span>
    <ul style="font-size: 0.90rem; margin-top: 10px; padding-left: 20px; opacity: 0.9;">
        <li><strong>40%</strong> Macro Stress Index</li>
        <li><strong>40%</strong> Week-of-Month Seasonality</li>
        <li><strong>20%</strong> Day-of-Week Seasonality</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)


with st.expander("The Math: Bayesian Priors, Custom Bounds, & Optimization"):
    st.markdown("""
    **1. Bayesian Stickiness Priors (Fixing the Flickering Problem)**
    Standard Maximum Likelihood Estimation (MLE) tends to over-optimize for local noise, causing the HMM to violently "flicker" between states intraday. To force the model to respect actual institutional timeframes, I injected a custom Bayesian penalty directly into the diagonal of the transition matrix during optimization:
    * **Macro Stickiness Penalty:** `100.0`
    * **Meso Stickiness Penalty:** `150.0`
    * **Micro Stickiness Penalty:** `250.0`
    By mathematically penalizing state-switching, the optimizer is forced to demand overwhelming statistical evidence before triggering a regime change.

    **2. Strict L-BFGS-B Optimization Constraints**
    Standard Python libraries often fail to converge on noisy FX data because they allow parameters to wander into mathematically impossible zones. I wrote a custom C-compiled objective function constrained by strict boundaries:
    * **Covariance Stationarity:** We strictly force $\\alpha + \\beta < 0.999$. If this bound is breached, the model assumes volatility will explode to infinity, completely breaking the GARCH forecasts.
    * **Finite 4th Moments:** The Hansen's Skew-t degrees of freedom ($\\nu$) is hard-bounded to $> 5.0$. This guarantees the math won't break when calculating kurtosis during "Black Swan" tail events.

    **3. Solving Path-Dependence (Haas et al. 2004)**
    If variance today depends on variance yesterday, and yesterday's variance depends on yesterday's regime, calculating all possible paths over 10 years becomes computationally impossible. Following Haas (2004), we bypass this by running independent GARCH(1,1) tracks completely parallel *inside* each regime:
    """)
    st.latex(r"\sigma_{t,i}^2 = \omega_i + \alpha_i (r_{t-1} - \mu_i - \phi_i r_{t-2})^2 + \beta_i \sigma_{t-1,i}^2")


# ==========================================
# SECTION 2: MODEL CONFIDENCE CHECKS
# ==========================================
st.markdown("<h2 id='2-mrm'><span class='section-badge'>Phase 2</span><span class='h2-text'> Model Confidence & Baseline Benchmarking</span></h2>", unsafe_allow_html=True)

st.markdown("If the model isn't highly confident about what regime the market is currently in, it shouldn't trade. We ran formal Model Risk checks to measure state clarity and benchmarked our complex HMM against a simple, standard GARCH model.")

col_mrm1, col_mrm2 = st.columns([1, 1.2])

with col_mrm1:
    st.markdown("#### Formal Health Checks")
    st.dataframe(get_mrm_audit(), use_container_width=True, hide_index=True)
    
with col_mrm2:
    st.markdown("""
<div class="audit-box" style="margin-top:0;">
<strong>RCM Clarity: Signal vs. Noise</strong><br>
The Regime Classification Measure (RCM) scored <strong>51.60%</strong>. For a 3-regime model, pure random guessing (33.3% probability per state) yields an RCM of 0, while perfect certainty yields 100. <br><br>
<strong>The Takeaway:</strong> A 51.6% score confirms the model <em>is</em> finding real, non-random structural patterns. However, it falls short of the >75% high-conviction target required for continuous trading. Because the 1-hour asset is so inherently noisy, the model's confidence fluctuates, leading to state-flipping that will heavily impact transaction fees.
</div>
""", unsafe_allow_html=True)

st.markdown("#### Probability Calibration (Brier Scores)")
st.markdown("While the RCM tells us how *decisive* the model is, the Brier Score tells us if its probability math is actually *accurate*. A Brier score ranges from 0 (perfect probability calibration) to 1 (completely wrong).")

col_cal1, col_cal2 = st.columns([1, 1.5])
with col_cal1:
    st.dataframe(get_calibration_data(), use_container_width=True, hide_index=True)
with col_cal2:
    st.markdown("""
<div class="insight-box" style="margin-top: 0;">
<strong>Well-Calibrated Uncertainty:</strong><br>
A global Brier score of 0.0702 is excellent. It proves that when the model says there is a 50% chance of a crisis, the market <em>actually is</em> in a 50/50 state of structural ambiguity. The low RCM isn't a failure of the math; it is a perfectly accurate reflection of EUR/USD's noisy intraday reality.
</div>
    """, unsafe_allow_html=True)

st.markdown("#### Benchmarking HMM Implied Volatility vs. Causal GARCH")
st.markdown("""To see if the advanced econometric engine actually added value, we benchmarked its volatility forecasts against a basic, dynamically-fitted GARCH(1,1) model.""")

st.dataframe(get_vol_benchmark(), use_container_width=True, hide_index=True)

st.image("vol_forecast_scatter.png", caption="MS-GARCH vs Causal GARCH Volatility Forecast Scatter", use_container_width=True)

st.markdown("""
<div class="insight-box">
<strong>Why use an HMM if a standard GARCH is just as accurate at forecasting?</strong><br>
A standard GARCH model is purely a scalar—it simply tells you <em>"volatility will be 12% tomorrow."</em> It doesn't tell you <em>why</em>, nor does it tell you how market physics have changed. <br><br>
The MS-GARCH engine identifies the actual structural environment. While a basic GARCH is fine for dynamically scaling position sizes, the HMM gives us the structural map required to completely swap out our underlying trading algorithms (e.g., turning off Mean Reversion and turning on Momentum when transitioning into a turbulent state).
</div>
""", unsafe_allow_html=True)


# ==========================================
# SECTION 3: REGIME DIAGNOSTICS
# ==========================================
st.markdown("<h2 id='3-diagnostics'><span class='section-badge'>Phase 3</span><span class='h2-text'> Regime Diagnostics</span></h2>", unsafe_allow_html=True)

st.markdown("""
Before using these regimes in a trading strategy, we have to prove they are statistically valid. We ran standard sanity checks on the out-of-sample data to ensure the model isn't just overfitting to historical noise.
""")


st.markdown("#### Out-of-Sample Sanity Checks")
st.dataframe(get_regime_diagnostics(), use_container_width=True, hide_index=True)
st.markdown("""
<div class="insight-box">
<strong>Statistical Validation:</strong><br>
<strong>Distributional Purity:</strong> Out-of-sample Kolmogorov-Smirnov (KS) tests yielded p-values near zero ($1.26 \\times 10^{-154}$), proving the regimes are not arbitrary splits but distinct statistical distributions. <br><br>
<strong>Tail Isolation:</strong> The model successfully partitioned the "fat tails." The 99% Expected Shortfall (CVaR-99) for the 'Crisis' regime is -48.8 bps, nearly double the 'Calm' regime's risk substrate.<br><br>
<strong>Limitation:</strong> The Event Study shows state transitions are <em>lagging indicators</em> due to the Hamilton Filter's smoothing delay. The model confirms volatility structural breaks rather than predicting them.
</div>
""", unsafe_allow_html=True)

st.image("oracle_diagnostics_dashboard.png", caption="Diagnostics Dashboard: KDE Distributional Purity & Event Study Trajectories", use_container_width=True)

# ==========================================
# SECTION 4: MULTI-TIMEFRAME INTEGRATION
# ==========================================
st.markdown("<h2 id='4-allocation'><span class='section-badge'>Phase 4</span><span class='h2-text'> Multi-Timeframe Integration (The 27-State Model)</span></h2>", unsafe_allow_html=True)

st.markdown("""
Looking at a single timeframe misses the broader market context. By nesting the Daily, 4-Hour, and 1-Hour models together, we created **27 unique market environments** ($3 \\times 3 \\times 3 = 27$). We trained 27 separate linear models (`RidgeCV`) on the out-of-sample data to act as "specialist experts" for each specific environment.

**Preventing Overfitting (L2 Regularization):** Slicing data into 27 micro-states creates small sample sizes, which is a massive curve-fitting risk. We used L2 penalty constraints to mathematically force noisy, fake signals to shrink to zero, ensuring our expert models only trade on robust, persistent factors.

**Continuous Soft-Routing:** Instead of using brittle IF/THEN rules (e.g., "If Crisis > 50%, turn off the strategy"), we dynamically blend the predictions from all 27 models based on real-time probabilities. This continuous blending prevents the portfolio from wildly whip-sawing at regime boundaries.
""")

col_ten1, col_ten2 = st.columns([1.3, 1])

with col_ten1:
    st.markdown("#### Performance Across the 27 Environments")
    st.dataframe(get_tensor_3d_data(), use_container_width=True, hide_index=True)
    
    
    st.markdown("#### Top 6 High-Alpha Micro-States")
    st.dataframe(get_top_10_states(), use_container_width=True, hide_index=True)

with col_ten2:
    st.markdown("#### Feature Importance (Top 10 Drivers)")
    st.dataframe(get_feature_ablation().head(10), use_container_width=True, hide_index=True)
    
    st.markdown("""
    <div class="insight-box">
    <strong>Calculus of Transition: Hazard Delta</strong><br>
    The most significant econometric feature was <code>Hazard_Delta</code>—the first derivative ($\Delta P / \Delta t$) of the Forward 4H Crisis probability. The model learned that <strong>acceleration</strong> in the crisis hazard rate is a higher-conviction signal for risk-off positioning than the absolute probability level itself.
    </div>
    """, unsafe_allow_html=True)

st.markdown("#### The Disagreement Filter: Do timeframes need to align?")
st.markdown("""
Most trend-following strategies wait for all timeframes to perfectly align before taking a trade. We tested this assumption by splitting our out-of-sample P&L based on how much the 1D, 4H, and 1H models disagreed with each other:
- **Low Disagreement (Timeframes Aligned):** Sharpe -0.32 | Ann. Return: -1.75%
- **High Disagreement (Timeframes Conflicting):** Sharpe -0.26 | Ann. Return: -1.96%

*Quant Insight:* For EUR/USD, the theoretical edge actually survives better during transitional, conflicting periods. When all three timeframes are perfectly aligned in a "Calm" state, the market is simply too efficient and trend-following fails.
""")


# ==========================================
# SECTION 5: META-REGIME ATTRIBUTION
# ==========================================
st.markdown("<h2 id='5-meta-regimes'><span class='section-badge'>Phase 5</span><span class='h2-text'> Meta-Regime Attribution</span></h2>", unsafe_allow_html=True)

st.markdown("""
Beyond the three primary regimes (Calm, Turbulent, Crisis), the 3D Volatility Tensor allows us to identify six "meta-regimes" that describe the structural interaction between the macro, meso, and micro environments. These are research constructs used to attribute exactly where the market is statistically inefficient.
""")

col_scen_img, col_scen_data = st.columns([1.2, 1])

with col_scen_img:
    st.image("scenario_and_regime_lift_dashboard.png", caption="Visualizing Meta-Regime Distributions", use_container_width=True)

with col_scen_data:
    st.markdown("#### Structural Inefficiency by Meta-Regime")
    st.dataframe(get_meta_regime_data(), use_container_width=True, hide_index=True)
    st.markdown("""
    <p style='font-size:0.95rem; opacity:0.8;'><em>Note: These meta-regimes are not traded strategies; they isolate specific structural alignments to measure where the model's directional logic is most accurate.</em></p>
    """, unsafe_allow_html=True)

col_scen1, col_scen2 = st.columns(2)

with col_scen1:
    st.markdown("""
    <div class="insight-box" style="margin-top:0;">
    <strong>The "Micro Shock" Environment</strong><br>
    The data shows that when intraday volatility spikes occur against a relatively stable macro backdrop (a "Micro Shock"), the model captures its highest directional edge (IC: +0.0332). This confirms theoretically that short-term mean-reversion overshoots are highly predictable when the larger macro state is not in distress.
    </div>
    """, unsafe_allow_html=True)

with col_scen2:
    st.markdown("""
    <div class="insight-box" style="margin-top:0; border-left-color: #2563EB !important;">
    <strong style="color:#2563EB;">The IC-to-Sharpe Divergence</strong><br>
    The <em>Mixed/Transitional</em> environment reveals a crucial structural tension. It possesses a high positive IC (+0.0349), meaning the model correctly predicts market direction. Yet, it generates negative P&L (-8.74%). Why? Because transitional periods generate extreme regime-switching activity, causing transaction friction to consume the mathematical edge.
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ==========================================
# SECTION 6: THEORETICAL & STRUCTURAL LIMITATIONS
# ==========================================
st.markdown("<h2 id='6-theoretical-limitations'><span class='section-badge'>Phase 6</span><span class='h2-text'> Theoretical & Structural Limitations</span></h2>", unsafe_allow_html=True)

st.markdown("""
An honest quantitative investigation must explicitly document where the underlying mathematics break down. Two of our formal Oracle Diagnostic tests failed, pointing to genuine limitations in describing EUR/USD volatility with a pure 3-state Markov sequence.
""")

col_tca1, col_tca2 = st.columns([1, 1])

with col_tca1:
    st.markdown("#### 1. Incomplete ACF Absorption")
    st.markdown("""
    <div class="audit-box" style="margin-top:10px; border-left-color: #EF4444 !important;">
    <strong>The Calm Regime Sub-Structure</strong><br>
    If a regime model correctly captures volatility clustering, the autocorrelation (ACF) of absolute returns <em>within</em> a specific regime should be lower than the unconditional baseline. <br><br>
    <strong>The Failure:</strong> The Calm regime's ACF (0.2625) was actually <em>higher</em> than the unconditional ACF (0.2579). <br><br>
    <strong>Interpretation:</strong> By isolating the 'Calm' observations, we inadvertently concentrated residual clustering. This proves the Calm state contains its own hidden sub-regimes (brief spikes interspersed with dead periods) that a single within-regime GARCH(1,1) specification cannot mathematically explain.
    </div>
    """, unsafe_allow_html=True)

with col_tca2:
    st.markdown("#### 2. The Early Warning Failure")
    st.markdown("""
    <div class="audit-box" style="margin-top:10px; border-left-color: #F59E0B !important;">
    <strong>Coincident vs. Leading Indicators</strong><br>
    We tested whether the crisis hazard probabilities could provide an early warning of an impending market drawdown in the subsequent 12 hours.<br><br>
    <strong>The Failure:</strong> The forward drawdown in the 12H following a hazard spike was statistically insignificant (-0.18%). <br><br>
    <strong>Interpretation:</strong> The HMM detects crises exactly as they occur, not before. The mathematics are reactive. The model cannot be used to pre-position for a structural break; it can only condition risk sizing <em>after</em> the break has been mathematically confirmed.
    </div>
    """, unsafe_allow_html=True)

st.markdown("#### The Transaction Cost Boundary (Why it is not an Alpha Engine)")
st.markdown("""
<p style="font-size:1.15rem; opacity:0.9; margin-bottom: 25px;">
While the master directional signal carries a persistent positive Information Coefficient (Raw IC: +0.0278, p=1.1e-05), the model is acutely sensitive to microstructure friction. A sensitivity analysis revealed that adding just <strong>+0.5bps of additional spread</strong> collapses the net return from -1.59% to -23.13%. High-frequency regime-switching generates valid statistical signals, but they are consumed by the friction of crossing the spread.
</p>
""", unsafe_allow_html=True)

col_ic_img1, col_ic_img2 = st.columns([1, 1.5])
with col_ic_img1:
    st.image("rolling_ic_stability.png", caption="Rank Correlation remains positive, but requires low-friction execution.", use_container_width=True)
with col_ic_img2:
    st.image("institutional_6_panel_wfa.png", caption="Visualizing the transition frequencies that cause execution drag.", use_container_width=True)

st.divider()

# ==========================================
# SECTION 7: FINAL VERDICT
# ==========================================
st.markdown("<h2 id='7-conclusion'><span class='section-badge'>Phase 7</span><span class='h2-text'> Final Verdict & Recommendations</span></h2>", unsafe_allow_html=True)

st.markdown("""
<div class="findings-box" style="border-left-color: #4B5563 !important; margin-bottom: 40px; margin-top: 0px;">
<div class="findings-title" style="color: #4B5563;">Final Hypotheses Evaluated</div>
<p style="margin-bottom:15px; color: var(--text-color); opacity: 0.95;">All findings are backed by formal statistical tests evaluated strictly out-of-sample. In institutional research, identifying boundary conditions is just as critical as proving econometric edge.</p>
<ul style="font-size:1.05rem;">
<li><strong>H1 (Regime filtering generates standalone execution alpha):</strong> 
    <br><span><span style="color:#EF4444; font-weight:bold;">[REJECTED]</span> While structural regimes exist, regime transitions are too frequent at the micro level. The friction of continuous state-switching mathematically destroys the theoretical execution edge.</span></li>
<li style="margin-top: 12px;"><strong>H2 (MS-GARCH Volatility Forecasting Superiority):</strong> 
    <br><span><span style="color:#10B981; font-weight:bold;">[VERIFIED]</span> The Diebold-Mariano test confirms MS-GARCH produces statistically superior volatility forecasts compared to Causal GARCH (DM-Stat +2.5790, p=0.00495). The multi-state architecture successfully isolates structural tail-risk.</span></li>
<li style="margin-top: 12px;"><strong>H3 (High entropy/noise strictly destroys predictive edge):</strong> 
    <br><span><span style="color:#EF4444; font-weight:bold;">[REJECTED]</span> By measuring Shannon Entropy, we found that the highest predictive rank correlation actually survived better during high-noise, transitional environments rather than pure calm ones.</span></li>
<li style="margin-top: 12px;"><strong>H4 (The Underlying Engine Predicts Direction):</strong> 
    <br><span><span style="color:#10B981; font-weight:bold;">[VERIFIED]</span> The tensor-routed master signal carries a persistent positive Information Coefficient (Raw IC: +0.0278 | p=1.1023e-05). The underlying directional physics are statistically sound.</span></li>
</ul>
<hr style="border-color: rgba(128,128,128,0.2); margin: 20px 0;">
<div style="font-size: 1.1rem;">
<strong>Senior Quant Conclusion:</strong><br>
This investigation proves that EUR/USD volatility is genuinely regime-structured, and that MS-GARCH provides a statistically superior fit to the data-generating process. However, the failure of the ACF absorption test and the reactive nature of the HMM transitions prove that regime models should not be used as standalone high-frequency execution engines. Their true institutional value lies in <strong>contextual risk filtering</strong>—scaling exposures based on verified structural state clarity.
</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style='text-align: center; margin-top: 50px; padding-top: 20px; border-top: 1px solid rgba(128, 128, 128, 0.2);'>
    <p style='font-size: 1.15rem; color: #6B7280; font-style: italic;'>
        If you made it all the way to the end, thank you for viewing my work. <br>
        I am always looking to refine these projects, so if you have critiques, suggestions, or just want to talk market dynamics, I'd love to hear them:
    </p>
    <a href='mailto:jayeshchaudharyofficial@gmail.com' style='font-size: 1.15rem; font-weight: 700; color: #FFFFFF; background-color: #3B82F6; padding: 10px 24px; border-radius: 6px; text-decoration: none; display: inline-block; transition: all 0.2s;'>
        ✉️ Email Me
    </a>
</div>
""", unsafe_allow_html=True)