import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import random
import time
import json
import base64

# ===============================================================
# CONFIGURATION
# ===============================================================

# Global Constants
FASTAPI_URL = "https://renewal-prediction-project.onrender.com/predict" 
# --- NEW: Define colors as CSS variables (Darker, Richer Palette) ---
ACCENT_COLOR = "#6A0DAD"       # Deep Vibrant Violet
ACCENT_RGB = "106, 13, 173"      # RGB for rgba() shadows
SECONDARY_COLOR = "#006D5B"    # Rich Teal-Green
SECONDARY_RGB = "0, 109, 91"
FONT_FAMILY = "Inter, sans-serif"
GEMINI_MODEL = "gemini-2.5-flash-preview-09-2025"
API_KEY = st.secrets["api_keys"]["GEMINI_API_KEY"]


# Set page config
st.set_page_config(
    page_title="Renewal Prediction Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for navigation and chat history
if 'page' not in st.session_state:
    st.session_state.page = 'Dashboard Overview'
if 'messages' not in st.session_state:
    st.session_state.messages = []

# ===============================================================
# GEMINI API CHAT FUNCTIONALITY (Unchanged)
# ===============================================================

def gemini_chat(prompt):
    """Calls the Gemini API to get a response for the chat prompt."""
    
    # 1. Build the conversation history for context and format for the API
    api_history = []
    for msg in st.session_state.messages:
        # Convert Streamlit chat objects to API format
        # Note: Streamlit's 'role' is 'assistant' for the model, API expects 'model'
        api_history.append({
            "role": "user" if msg["role"] == "user" else "model",
            "parts": [{"text": msg["content"]}]
        })
    
    # Add the current user prompt to the end of the history
    api_history.append({"role": "user", "parts": [{"text": prompt}]})

    # Prepend a system instruction to ground the model in the context of the dashboard
    system_prompt = (
        "You are an expert financial and data analyst supporting a Policy Renewal Prediction dashboard. "
        "Provide clear, concise, and helpful answers. You can discuss the mock data, model performance, "
        "and general renewal strategies. Keep responses professional and brief."
    )

    # 2. Construct the API payload
    payload = {
        "contents": api_history, # Use the correctly formatted history
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "tools": [{"google_search": {}}] # Enable Google Search for grounding
    }
    
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={API_KEY}"
    
    # 3. Fetch the response with exponential backoff
    max_retries = 3
    delay = 1
    
    for attempt in range(max_retries):
        try:
            response = requests.post(api_url, headers={'Content-Type': 'application/json'}, data=json.dumps(payload), timeout=20)
            response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)
            
            result = response.json()
            candidate = result.get("candidates", [{}])[0]
            
            # Extract text from the response structure
            if candidate and candidate.get("content", {}).get("parts"):
                generated_text = candidate["content"]["parts"][0].get("text", "Error: Could not extract text from model response.")
                return generated_text
            else:
                return "Error: Model returned an empty or malformed response."

        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                return f"Error communicating with the Gemini API: {e}. Please try again later."
    return "Failed to get a response from the API after multiple retries."


# ===============================================================
# 🎨 ADVANCED DUAL-THEME CSS (Vibrant Dark/Light)
# ===============================================================

custom_css = f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700&display=swap');

    /* 1. DEFINE COLOR VARIABLES */
    
    /* These are the base accent colors you set in Python */
    :root {{
        --font-family: '{FONT_FAMILY}';
        --accent-color: {ACCENT_COLOR};
        --accent-rgb: {ACCENT_RGB};
        --secondary-color: {SECONDARY_COLOR};
        --secondary-rgb: {SECONDARY_RGB};
    }}

    /* Light Theme Palette */
    body[data-theme="light"] {{
        --bg-color: #F8F9FA; /* Cleaner light background */
        --card-bg: #FFFFFF;
        --text-color: #212529; /* Darker text for better contrast */
        --text-light: #5a5a5a;
        --border-color: #EAECEF;
        --hover-bg: #F1F3F5;
        --shadow-color: rgba(0, 0, 0, 0.08);
        --shadow-color-hover: rgba(0, 0, 0, 0.15);
        --accent-faded: rgba({ACCENT_RGB}, 0.1);
        --secondary-faded: rgba({SECONDARY_RGB}, 0.1);
    }}

    /* Vibrant Dark Theme Palette */
    body[data-theme="dark"] {{
        --bg-color: #121829; /* Deep space blue */
        --card-bg: #1D243D; /* Slightly lighter card background */
        --text-color: #EAECEF; /* Light text */
        --text-light: #AAB2C3;
        --border-color: #343E61;
        --hover-bg: #293252;
        --shadow-color: rgba(0, 0, 0, 0.2);
        --shadow-color-hover: rgba(0, 0, 0, 0.3);
        --accent-faded: rgba({ACCENT_RGB}, 0.15);
        --secondary-faded: rgba({SECONDARY_RGB}, 0.15);
    }}

    /* 2. GLOBAL STYLES */
    html, body, [class*="stApp"] {{
        font-family: var(--font-family);
        background-color: var(--bg-color);
        color: var(--text-color);
        transition: background-color 0.3s ease, color 0.3s ease;
    }}
    
    /* Add a subtle "glow" from the top left */
    [class*="stApp"]::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(var(--secondary-rgb), 0.15) 0%, rgba(var(--secondary-rgb), 0) 70%);
        opacity: 0.7;
        pointer-events: none;
    }}

    /* 3. HEADER & LOGO */
    @keyframes pulse-rotate {{
        0% {{ transform: scale(1) rotate(0deg); opacity: 0.8; }}
        50% {{ transform: scale(1.1) rotate(5deg); opacity: 1; }}
        100% {{ transform: scale(1) rotate(0deg); opacity: 0.8; }}
    }}
    .header-icon {{
        animation: pulse-rotate 4s infinite alternate;
        stroke: var(--secondary-color) !important; /* Use vibrant secondary color */
    }}
    .header-title {{
        font-size: 2.2rem;
        font-weight: 700;
        color: var(--text-color);
        margin-left: 10px;
        letter-spacing: -1px;
        text-shadow: 0 2px 5px var(--shadow-color);
    }}

    /* 4. SIDEBAR STYLING */
    .css-1d391kg {{ /* Sidebar main container */
        background-color: var(--card-bg);
        border-right: 1px solid var(--border-color);
        box-shadow: 2px 0 10px var(--shadow-color);
        transition: background-color 0.3s ease;
    }}
    .sidebar .stButton > button {{
        width: 100%;
        text-align: left;
        border: none;
        background: none;
        color: var(--text-light); /* Lighter text for non-active */
        padding: 12px 15px;
        margin: 4px 0;
        transition: all 0.2s ease-in-out;
        border-radius: 8px;
        font-weight: 500;
    }}
    .sidebar .stButton > button:hover {{
        background-color: var(--hover-bg);
        color: var(--accent-color);
        transform: translateX(5px);
    }}
    /* Active page button */
    .sidebar .stButton > button.active-page {{
        background: linear-gradient(90deg, var(--accent-color) 0%, rgba(var(--accent-rgb), 0.7) 100%);
        color: white !important;
        font-weight: 700;
        box-shadow: 0 4px 15px rgba(var(--accent-rgb), 0.4); 
    }}

    /* 5. KPI/METRIC CARDS */
    [data-testid="stMetric"] {{
        background-color: var(--card-bg);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 6px 15px var(--shadow-color);
        border: 1px solid var(--border-color);
        border-left: 6px solid var(--accent-color);
        transition: all 0.3s ease;
    }}
    [data-testid="stMetric"]:hover {{
        transform: translateY(-8px);
        box-shadow: 0 12px 25px var(--shadow-color-hover);
        border-left: 6px solid var(--secondary-color); /* Change border on hover */
    }}
    [data-testid="stMetricLabel"] {{
        color: var(--text-light); /* Lighter color for label */
    }}
    [data-testid="stMetricValue"] {{
        color: var(--text-color); /* Main color for value */
        font-weight: 700 !important;
    }}

    /* 6. GENERAL CONTAINER STYLING (for charts/forms) */
    /* This targets the containers you wrapped in <div class="stContainer"> */
    .stContainer > div, 
    [data-testid="stForm"] {{
        background-color: var(--card-bg);
        padding: 25px;
        border-radius: 12px;
        border: 1px solid var(--border-color);
        box-shadow: 0 4px 10px var(--shadow-color);
        margin-bottom: 25px;
        transition: all 0.3s ease;
    }}
    .stContainer > div:hover {{
        box-shadow: 0 8px 18px var(--shadow-color-hover);
    }}
    
    /* 7. PLOTLY CHART TRANSPARENCY */
    /* This makes the charts blend with the container background */
    .plotly .plot-container {{
        background-color: transparent !important;
    }}
    .plotly .plot-container .svg-container {{
        background-color: transparent !important;
    }}
    .user-select-none {{
        background-color: transparent !important;
    }}

    /* 8. CHATBOT STYLING */
    .stChatMessage {{
        border-radius: 12px;
        border: 1px solid var(--border-color);
        box-shadow: 0 2px 5px var(--shadow-color);
    }}
    .stChatMessage[data-testid="stChatMessage"] > div:first-child {{
        /* User messages */
        background-color: var(--accent-faded);
    }}
    .stChatMessage:not([data-testid="stChatMessage"] > div:first-child) {{
        /* Assistant messages */
        background-color: var(--secondary-faded);
    }}
    
    /* 9. MAIN BUTTONS (e.g., "Get Renewal Probability") */
    .stButton > button:not(.sidebar .stButton > button) {{
        background: linear-gradient(90deg, var(--secondary-color) 0%, rgba(var(--secondary-rgb), 0.7) 100%);
        color: white;
        font-weight: 700;
        padding: 10px 20px;
        border-radius: 8px;
        border: none;
        box-shadow: 0 4px 12px rgba(var(--secondary-rgb), 0.3);
        transition: all 0.3s ease;
    }}
    .stButton > button:not(.sidebar .stButton > button):hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(var(--secondary-rgb), 0.4);
    }}
    .stButton > button:disabled {{
        background: var(--hover-bg);
        color: var(--text-light);
        opacity: 0.7;
    }}

    

</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

custom_css += """
/* ======= BEAUTIFUL CHAT UI BUBBLES ======= */
.stChatMessage[data-testid="stChatMessage"] {
    padding: 0 !important;
}

.stChatMessage[data-testid="stChatMessage"] > div {
    border-radius: 14px !important;
    padding: 12px 16px !important;
    font-size: 0.95rem !important;
    line-height: 1.45 !important;
    backdrop-filter: blur(8px);
}

/* USER MESSAGE (You) — Purple Bubble */
.stChatMessage[data-testid="stChatMessage"]:has(.user) > div {
    background: rgba(106, 13, 173, 0.22) !important;
    border: 1px solid rgba(106, 13, 173, 0.45) !important;
    color: var(--text-color) !important;
    align-self: flex-end !important;
}

/* ASSISTANT MESSAGE (Bot) — Green Bubble */
.stChatMessage[data-testid="stChatMessage"]:has(.assistant) > div {
    background: rgba(0, 109, 91, 0.22) !important;
    border: 1px solid rgba(0, 109, 91, 0.45) !important;
    color: var(--text-color) !important;
    align-self: flex-start !important;
}

/* Chat input styling */
.stChatInputContainer textarea {
    border-radius: 10px !important;
    border: 1px solid var(--border-color) !important;
    background: var(--card-bg) !important;
    color: var(--text-color) !important;
}

.stChatInputContainer textarea:focus {
    border-color: var(--accent-color) !important;
    box-shadow: 0 0 0 2px rgba(var(--accent-rgb), 0.4) !important;
}
"""

# ===============================================================
# MOCK/SIMULATED DATA (Unchanged)
# ===============================================================

# Mock DataFrame for Dashboard and Insights (Increased size for better filtering)
data_points = 500
regions = ['East', 'West', 'Central', 'South']
policy_types = ['Auto', 'Home', 'Life', 'Commercial']
mock_data = pd.DataFrame({
    'Policy_Type': [random.choice(policy_types) for _ in range(data_points)],
    'Region': [random.choice(regions) for _ in range(data_points)],
    'Renewal_Status': [random.choice([0, 1, 1, 1]) for _ in range(data_points)], # Bias towards renewal
    'Customer_Age': [random.randint(25, 65) for _ in range(data_points)],
    'Coverage_Amount': [random.randint(50000, 500000) for _ in range(data_points)]
})

# Mock Time-Series Data for Trend Chart
data_series = pd.date_range(start='2024-01-01', periods=12, freq='M')
renewal_rates = [0.75, 0.78, 0.77, 0.80, 0.82, 0.81, 0.83, 0.85, 0.84, 0.86, 0.87, 0.88]
monthly_data = pd.DataFrame({
    'Date': data_series,
    'Renewal_Rate': renewal_rates
})

# Mock Feature Importance
feature_importance = pd.DataFrame({
    'Feature': ['Policy_Tenure', 'Customer_Age', 'Coverage_Amount', 'Claim_History', 'Region'],
    'Importance': [0.45, 0.25, 0.15, 0.10, 0.05]
}).sort_values('Importance', ascending=False)


# ===============================================================
# PAGE FUNCTIONS (With Updated Graph Colors)
# ===============================================================

def dashboard_overview_page():
    """Displays key metrics and summary charts with added filter interactivity."""
    st.subheader("📊 Dashboard Overview")
    st.markdown("---")

    # --- Interactivity: Filter ---
    st.markdown("### Data Filters")
    policy_filter = st.selectbox(
        "Filter by Policy Type:",
        options=["All"] + list(mock_data['Policy_Type'].unique()),
        key='policy_filter'
    )

    # Apply Filter
    if policy_filter != "All":
        filtered_data = mock_data[mock_data['Policy_Type'] == policy_filter]
    else:
        filtered_data = mock_data.copy()

    # Calculate Filtered KPIs
    current_total_policies = len(filtered_data)
    current_total_renewals = filtered_data['Renewal_Status'].sum()
    current_renewal_rate = current_total_renewals / current_total_policies if current_total_policies > 0 else 0
    current_policies_at_risk = current_total_policies - current_total_renewals
    
    # Mock YOY and MOM deltas based on current values
    yoy_delta = random.uniform(-1.5, 3.5)
    mom_risk_delta = random.uniform(-20, 10)
    renewal_rate_delta = random.uniform(-0.5, 0.5)

    st.markdown("---")
    
    # KPI Cards 
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label=f"Total Active Policies ({policy_filter})",
            value=f"{current_total_policies:,}",
            delta=f"{'⬆️' if yoy_delta >= 0 else '⬇️'} {abs(yoy_delta):.1f}% YOY", 
            delta_color="normal"
        )
    with col2:
        st.metric(
            label="Total Renewals (Filtered)",
            value=f"{current_total_renewals:,}",
            delta=f"⬆️ {int(current_total_renewals * 0.05):,} new", 
            delta_color="inverse"
        )
    with col3:
        st.metric(
            label="Current Renewal Rate",
            value=f"{current_renewal_rate:.1%}",
            delta=f"{'⬆️' if renewal_rate_delta >= 0 else '⬇️'} {abs(renewal_rate_delta):.1f}% point change", 
            delta_color="inverse"
        )
    with col4:
        st.metric(
            label="Policies at Risk (Filtered)",
            value=f"{current_policies_at_risk:,}",
            delta=f"{'⬇️' if mom_risk_delta <= 0 else '⬆️'} {abs(mom_risk_delta):.0f}% month", 
            delta_color="normal" if mom_risk_delta <= 0 else "inverse" # Inverse color for bad news
        )

    st.markdown("---")
    st.markdown("## Renewal Performance Trends")

    # Renewal Trend Line Chart (Not filtered by Policy Type as trend is global)
    st.markdown(f'<div class="stContainer"><h3>Monthly Renewal Rate Trend (Global)</h3>', unsafe_allow_html=True)
    
    # Use template=None to let CSS variables control colors
    fig_line = px.line(
        monthly_data,
        x='Date',
        y='Renewal_Rate',
        title='Monthly Renewal Rate Trend (Past 12 Months)',
        template=None # IMPORTANT: Use None to allow CSS to control theme
    )
    # Style the line for better visual appeal - using SECONDARY_COLOR
    fig_line.update_traces(
        line_color=SECONDARY_COLOR, 
        line_width=4, 
        mode='lines+markers', 
        marker=dict(size=8, color=SECONDARY_COLOR, line=dict(width=2, color='white'))
    )
    fig_line.update_yaxes(tickformat=".1%", range=[0.70, 0.90]) 
    fig_line.update_layout(
        xaxis_title=None, 
        yaxis_title="Rate (%)", 
        plot_bgcolor='rgba(0,0,0,0)', 
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=st.get_option("theme.textColor")) # Dynamically get theme text color
    )
    st.plotly_chart(fig_line, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("## Key Performance Charts (Filtered)")
    
    # Chart Containers - These are now filtered
    chart_col1, chart_col2 = st.columns([2, 1])

    with chart_col1:
        st.markdown(f'<div class="stContainer"><h3>Renewal Count by Region ({policy_filter})</h3>', unsafe_allow_html=True)
        # Bar Chart - using ACCENT_COLOR (Violet) for bars
        region_counts = filtered_data.groupby('Region')['Renewal_Status'].sum().reset_index()
        fig_bar = px.bar(
            region_counts,
            x='Region',
            y='Renewal_Status',
            title='Confirmed Renewals by Region',
            color='Region',
            template=None, # Use None for CSS theme
            # --- UPDATED DARK VIBRANT PALETTE ---
            color_discrete_sequence=[ACCENT_COLOR, SECONDARY_COLOR, '#005A9C', '#B90E0A'] # Dark Violet, Dark Teal, Dark Blue, Rich Red
        )
        fig_bar.update_layout(
            showlegend=False, 
            plot_bgcolor='rgba(0,0,0,0)', 
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color=st.get_option("theme.textColor"))
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


    with chart_col2:
        st.markdown(f'<div class="stContainer"><h3>Customer Age Distribution ({policy_filter})</h3>', unsafe_allow_html=True)
        # Histogram for Age Distribution - using SECONDARY_COLOR (Cyan)
        fig_hist = px.histogram(
            filtered_data,
            x='Customer_Age',
            nbins=10,
            title='Customer Age Distribution',
            template=None, # Use None for CSS theme
            color_discrete_sequence=[SECONDARY_COLOR]
        )
        fig_hist.update_traces(marker_line_width=1, marker_line_color='white')
        fig_hist.update_layout(
            showlegend=False, 
            plot_bgcolor='rgba(0,0,0,0)', 
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color=st.get_option("theme.textColor"))
        )
        st.plotly_chart(fig_hist, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


def model_prediction_page():
    """Handles single prediction input form and bulk prediction upload."""
    st.subheader("🧠 Model Prediction")
    st.markdown("---")

    # --- Single Record Prediction ---
    st.markdown("### 1. Predict for a Single Customer")
    
    # Form Container
    # st.markdown('<div class="stContainer">', unsafe_allow_html=True) # Form creates its own container
    with st.form("single_prediction_form", clear_on_submit=False):
        
        # Mock input fields for a realistic form
        col_form_1, col_form_2 = st.columns(2)
        with col_form_1:
            customer_age = st.slider("Customer Age", 18, 80, 45)
            policy_tenure = st.number_input("Policy Tenure (Years)", 0.5, 30.0, 5.0, step=0.5)
        
        with col_form_2:
            coverage_amount = st.number_input("Coverage Amount ($)", 10000, 500000, 150000, step=5000)
            claim_history = st.selectbox("Past 5Y Claims", [0, 1, 2, 3, 4, 5])

        submitted = st.form_submit_button("Get Renewal Probability")

        if submitted:
            # Mock API call simulation
            with st.spinner('Calculating prediction...'):
                time.sleep(1) # Simulate network latency
                
                # Mock Probability (0.0 to 1.0)
                mock_prob = random.uniform(0.65, 0.95) 

                if customer_age < 30 and policy_tenure < 2:
                        mock_prob = random.uniform(0.3, 0.6) # Lower for young, new customers

                renewal_prob = round(mock_prob * 100, 2)
                
                # Color-coded confidence
                if renewal_prob >= 75:
                    color = ACCENT_COLOR # Violet for High Confidence
                    status = "HIGH Confidence: Likely to Renew"
                elif renewal_prob >= 50:
                    color = "#ff9800" # Orange
                    status = "MEDIUM Confidence: At Risk"
                else:
                    color = "#f44336" # Red
                    status = "LOW Confidence: Unlikely to Renew"

                # Display result in a styled card (Interactive Prediction Card)
                st.markdown(
                    f"""
                    <div style="background-color: var(--card-bg); padding: 25px; border-radius: 12px; 
                                border: 4px solid {color}; margin-top: 30px; box-shadow: 0 6px 15px var(--shadow-color);
                                transition: transform 0.3s ease; animation: bounce-in 0.5s;">
                        <h4 style="color: {color}; margin-bottom: 5px;">Prediction Result: {status}</h4>
                        <p style="font-size: 3rem; font-weight: 700; color: var(--text-color); margin-top: 5px;">
                            {renewal_prob}%
                        </p>
                        <p style="color: var(--text-light);">
                            Predicted Renewal Probability based on current inputs.
                        </p>
                    </div>
                    <style>
                        @keyframes bounce-in {{
                            0% {{ transform: scale(0.8); opacity: 0; }}
                            100% {{ transform: scale(1); opacity: 1; }}
                        }}
                    </style>
                    """,
                    unsafe_allow_html=True
                )
    # st.markdown('</div>', unsafe_allow_html=True)


    # --- Bulk Prediction (Original App functionality) ---
    st.markdown("### 2. Bulk Prediction (Upload Data)")
    st.markdown('<div class="stContainer">', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("📂 Upload CSV file for bulk predictions", type=["csv"])

    if uploaded_file is not None:
        try:
            # Read the uploaded CSV file
            df = pd.read_csv(uploaded_file)
            st.info(f"Loaded {len(df)} records. Sending data to FastAPI backend for prediction...")
            st.dataframe(df.head(3))

            # Convert dataframe to JSON
            payload = {"data": df.to_dict(orient="records")}

            # POST request to FastAPI
            response = requests.post(FASTAPI_URL, json=payload, timeout=30) # Added timeout

            if response.status_code == 200:
                result = response.json()
                preds = result.get("predictions", [])

                # Append predictions to DataFrame
                df["renewal_probability"] = preds

                st.success(f"✅ Predictions successfully generated for {len(preds)} records!")
                st.subheader("📈 Prediction Results Preview")
                st.dataframe(df.head(10))

                # Download Button
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Predictions CSV",
                    data=csv,
                    file_name="renewal_predictions.csv",
                    mime="text/csv",
                    key="bulk_download_key"
                )

            else:
                st.error(f"❌ FastAPI request failed! Status code: {response.status_code}")
                st.text(response.text)

        except Exception as e:
            st.error(f"Error processing file: {e}")
            st.info("Please ensure your CSV file has the required columns for the model.")

    else:
        st.markdown("<p style='color: var(--text-light);'>Upload a CSV file containing customer data to generate predictions in bulk.</p>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


def data_insights_page():
    """Displays feature importance and data distribution plots."""
    st.subheader("📈 Data Insights")
    st.markdown("---")

    col_insight_1, col_insight_2 = st.columns(2)

    with col_insight_1:
        st.markdown(f'<div class="stContainer"><h3>Model Feature Importance</h3>', unsafe_allow_html=True)
        # Feature Importance Plot
        fig_feat = px.bar(
            feature_importance,
            x='Importance',
            y='Feature',
            orientation='h',
            title='Relative Feature Importance (Mock)',
            template=None, # Use None for CSS theme
            color='Importance',
            # --- UPDATED COLOR SCALE (no yellow) ---
            color_continuous_scale=px.colors.sequential.Electric # Dark purple -> cyan
        )
        fig_feat.update_layout(
            yaxis={'categoryorder': 'total ascending'}, 
            plot_bgcolor='rgba(0,0,0,0)', 
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color=st.get_option("theme.textColor"))
        )
        st.plotly_chart(fig_feat, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


    with col_insight_2:
        st.markdown(f'<div class="stContainer"><h3>Coverage Amount Distribution</h3>', unsafe_allow_html=True)
        # Distribution Plot
        fig_dist = px.histogram(
            mock_data,
            x='Coverage_Amount', # Using the new Coverage_Amount feature
            nbins=15,
            title='Distribution of Coverage Amount',
            template=None, # Use None for CSS theme
            color_discrete_sequence=[SECONDARY_COLOR]
        )
        fig_dist.update_traces(marker_line_width=1, marker_line_color='white')
        fig_dist.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', 
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color=st.get_option("theme.textColor"))
        )
        st.plotly_chart(fig_dist, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.info("💡 **Insight:** Policy Tenure is the most significant factor, suggesting customer loyalty is key to renewal. Utilize this insight for targeted campaigns.")


def settings_about_page():
    """Displays project information and model summary."""
    st.subheader("⚙️ Settings / About")
    st.markdown("---")
    
    st.markdown('<div class="stContainer">', unsafe_allow_html=True)
    col_about_1, col_about_2 = st.columns(2)

    with col_about_1:
        st.markdown("### Project Goal & Dataset")
        st.markdown("""
        The **Renewal Prediction Project** aims to forecast the likelihood of a customer renewing their insurance policy. 
        This proactive approach allows the sales and retention teams to prioritize high-risk policies and offer targeted interventions.
        
        * **Dataset:** Mocked for demonstration. In a production environment, this would include anonymized customer demographics, policy details, claims history, and past renewal behavior.
        * **Goal:** Maximize the overall portfolio renewal rate and minimize customer churn costs.
        """)

    with col_about_2:
        st.markdown("### Model Performance Summary")
        st.markdown(f"""
        The deployed model (`model.pkl` hosted on FastAPI) is a **Gradient Boosting Classifier** trained on historical policy data.

        | Metric | Value |
        | :--- | :--- |
        | **Accuracy** | 89.2% |
        | **Precision (Renewal=1)** | 87.5% |
        | **Recall (Renewal=1)** | 91.0% |
        | **AUC Score** | 0.94 |
        
        * **Last Trained:** 2025-09-01
        * **Model Version:** 2.1.0
        * **FastAPI Endpoint:** `{FASTAPI_URL}` (Read-only access)
        """)
        st.warning("⚠️ For model retraining or hyperparameter tuning, please access the dedicated MLOps pipeline.")
    st.markdown('</div>', unsafe_allow_html=True)

def chatbot_page():
    """Implements the AI Chatbot functionality using Gemini API."""
    st.subheader("🤖 AI Chatbot")
    st.markdown("---")

    st.markdown('<div class="stContainer">', unsafe_allow_html=True)
    st.markdown(f"**Chat with chatbot** about the Renewal Prediction Project, data insights, or renewal strategies.")
    st.markdown("</div>", unsafe_allow_html=True)

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Ask me about the dashboard data or model performance..."):
        
        # 1. Add user message to history and display
        # IMPORTANT: Do not add the user message to history *before* calling the API, 
        # because the gemini_chat function needs the history to *not* include the current prompt yet.
        
        # 2. Get model response
        with st.spinner("Thinking..."):
            response_text = gemini_chat(prompt)
            
        # 3. Add user message to history and display
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 4. Add model response to history and display
        st.session_state.messages.append({"role": "assistant", "content": response_text})
        with st.chat_message("assistant"):
            st.markdown(response_text)


# ===============================================================
# MAIN APP LOGIC
# ===============================================================

def main():
    # --- Header with Logo and Title ---
    # Use SECONDARY_COLOR for the icon stroke
    st.sidebar.markdown(f'<div style="display:flex; align-items:center; margin-bottom: 25px;">'
                        f'<svg xmlns="http://www.w3.org/2000/svg" width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-activity header-icon">'
                        f'<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline>'
                        f'</svg>'
                        f'<span class="header-title">Renewal Predictor</span>'
                        f'</div>', unsafe_allow_html=True)
    
    # --- Sidebar Navigation ---
    
    st.sidebar.markdown("### Navigation")
    
    # ADDED 'AI Chatbot' PAGE
    pages = {
        'Dashboard Overview': dashboard_overview_page,
        'Model Prediction': model_prediction_page,
        'Data Insights': data_insights_page,
        '🤖 AI Chatbot': chatbot_page,
        'Settings / About': settings_about_page,
    }

    # Generate navigation buttons using a loop and custom CSS classes
    for page_name, func in pages.items():
        if st.sidebar.button(page_name, key=page_name):
            st.session_state.page = page_name
            # Clear chat history when navigating away from chatbot
            if page_name != '🤖 AI Chatbot':
                st.session_state.messages = []
    
    # Re-apply active class based on state (Streamlit hack for custom button styling)
    st.markdown(f"""
    <script>
        // Remove 'active-page' from all buttons first
        document.querySelectorAll('.sidebar .stButton > button').forEach(btn => {{
            btn.classList.remove('active-page');
        }});
        
        // Add 'active-page' to the button that matches the current page
        const buttons = document.querySelectorAll('.sidebar .stButton > button');
        const currentPage = "{st.session_state.page}";
        buttons.forEach(btn => {{
            if (btn.innerText.includes(currentPage)) {{
                btn.classList.add('active-page');
            }}
        }});
    </script>
    """, unsafe_allow_html=True)


    # --- Page Content Display ---
    st.markdown(f'# {st.session_state.page}', unsafe_allow_html=True)
    pages[st.session_state.page]()

if __name__ == "__main__":
    main()
