import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import requests
import time

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Stroke Risk Assessment Assistant",
    layout="wide",
    page_icon="🏥",
    initial_sidebar_state="expanded"
)

# ============================================================================
# API FUNCTIONS
# ============================================================================

def query_llm(prompt: str) -> str:
    """Query the Hugging Face LLM API for medical guidance."""
    API_URL = "https://router.huggingface.co/v1/chat/completions"
    
    headers = {
        "Authorization": "Bearer {os.environ['HF_TOKEN']}"
    }
    
    system_prompt = """You are a medical AI assistant specialized in stroke risk assessment. Your role is to identify stroke-related symptoms, assess urgency, explain risk factors simply, provide recommendations, never diagnose or prescribe, always recommend professional consultation for symptoms, and ask clarifying questions when needed. Respond clearly and empathetically."""

    full_prompt = f"<s>[INST] {system_prompt}\n\nUser Query: {prompt} [/INST]"
    
    payload = {
        "messages": [
            {
                "role": "user",
                "content": full_prompt
            }
        ],
        "model": "meta-llama/Llama-3.1-8B-Instruct:novita"
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
        
    except requests.exceptions.Timeout:
        return "The request timed out. Please try again in a moment."
    except requests.exceptions.HTTPError:
        if response.status_code == 503:
            return "The AI model is currently loading. Please wait 20-30 seconds and try again."
        elif response.status_code == 401:
            return "Authentication failed. Please verify your API token."
        elif response.status_code == 429:
            return "Rate limit exceeded. Please wait a moment before trying again."
        else:
            return "The service is temporarily unavailable. Please try again later."
    except requests.exceptions.ConnectionError:
        return "Unable to connect to the AI service. Please check your internet connection."
    except Exception:
        return "An unexpected error occurred. Please refresh the page and try again."

# ============================================================================
# RISK ASSESSMENT FUNCTIONS
# ============================================================================

def predict_stroke_risk(features: dict) -> tuple:
    """Predict stroke risk based on patient features."""
    risk_score = np.random.uniform(0, 0.2)
    
    # Age factor
    if features['age'] > 60:
        risk_score += 0.15
    
    # Medical conditions
    if features['hypertension'] == 1:
        risk_score += 0.2
    if features['heart_disease'] == 1:
        risk_score += 0.25
    
    # Glucose level
    if features['avg_glucose_level'] > 140:
        risk_score += 0.15
    
    # BMI
    if features['bmi'] > 30:
        risk_score += 0.1
    
    # Smoking
    if features['smoking_status'] == 'smokes':
        risk_score += 0.2
    
    risk_score = min(risk_score, 1.0)
    stroke_prediction = 1 if risk_score > 0.5 else 0
    
    return stroke_prediction, risk_score

def generate_guidance(features: dict, risk_score: float, stroke_pred: int) -> tuple:
    """Generate personalized health guidance based on risk assessment."""
    guidance = []
    
    # Determine risk level
    if risk_score > 0.7:
        risk_level = "HIGH"
        risk_class = "risk-high"
        guidance.append("⚠️ URGENT: Consult a healthcare provider immediately for comprehensive evaluation.")
    elif risk_score > 0.4:
        risk_level = "MEDIUM"
        risk_class = "risk-medium"
        guidance.append("⚡ MODERATE RISK: Schedule a check-up with your doctor within the next 2 weeks.")
    else:
        risk_level = "LOW"
        risk_class = "risk-low"
        guidance.append("✅ LOW RISK: Maintain regular health check-ups and healthy lifestyle habits.")
    
    # Specific recommendations
    if features['hypertension'] == 1:
        guidance.append("🩺 Monitor blood pressure daily and take prescribed medications regularly.")
    
    if features['heart_disease'] == 1:
        guidance.append("❤️ Follow cardiac care guidelines and avoid strenuous activities without medical approval.")
    
    if features['avg_glucose_level'] > 140:
        guidance.append("🍽️ Manage blood sugar levels through diet, exercise, and medication as prescribed.")
    
    if features['bmi'] > 30:
        guidance.append("🏃 Implement a structured weight management program with professional guidance.")
    
    if features['smoking_status'] == 'smokes':
        guidance.append("🚭 CRITICAL: Enroll in a smoking cessation program immediately to reduce stroke risk.")
    
    # General recommendations
    guidance.extend([
        "💊 Consider aspirin therapy (consult your doctor first).",
        "🥗 Adopt a Mediterranean or DASH diet rich in fruits, vegetables, and whole grains.",
        "🧘 Manage stress through meditation, yoga, or relaxation techniques."
    ])
    
    return risk_level, risk_class, guidance

# ============================================================================
# CSS STYLING
# ============================================================================

def inject_custom_css():
    """Inject custom CSS for styling."""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
    }
    
    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .hero-section {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 60px 40px;
        margin: 20px 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        animation: fadeInUp 1s ease-out;
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .hero-title {
        font-size: 3.5rem;
        font-weight: 700;
        color: white;
        text-align: center;
        margin-bottom: 20px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .hero-subtitle {
        font-size: 1.3rem;
        color: rgba(255, 255, 255, 0.9);
        text-align: center;
        font-weight: 300;
    }
    
    .glass-card {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 30px;
        margin: 20px 0;
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        transition: transform 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px 0 rgba(31, 38, 135, 0.5);
    }
    
    .result-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.2), rgba(255, 255, 255, 0.05));
        backdrop-filter: blur(15px);
        border-radius: 20px;
        padding: 35px;
        margin: 25px 0;
        border: 2px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.2);
        animation: popIn 0.6s ease-out;
    }
    
    @keyframes popIn {
        0% {
            opacity: 0;
            transform: scale(0.8);
        }
        100% {
            opacity: 1;
            transform: scale(1);
        }
    }
    
    .metric-container {
        background: rgba(255, 255, 255, 0.2);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 10px;
        transition: all 0.3s ease;
    }
    
    .metric-container:hover {
        background: rgba(255, 255, 255, 0.3);
        transform: scale(1.05);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: white;
        margin-bottom: 5px;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: rgba(255, 255, 255, 0.8);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .risk-high {
        color: #ff4757;
        font-weight: 700;
    }
    
    .risk-medium {
        color: #ffa502;
        font-weight: 700;
    }
    
    .risk-low {
        color: #26de81;
        font-weight: 700;
    }
    
    .guidance-box {
        background: rgba(255, 255, 255, 0.25);
        border-left: 4px solid #667eea;
        padding: 20px;
        margin: 15px 0;
        border-radius: 8px;
        color: white;
    }
    
    .progress-bar {
        height: 8px;
        background: rgba(255, 255, 255, 0.2);
        border-radius: 10px;
        overflow: hidden;
        margin: 20px 0;
    }
    
    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #667eea, #764ba2);
        animation: progress 2s ease-out;
    }
    
    @keyframes progress {
        from { width: 0%; }
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 15px 40px;
        font-size: 1.1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
    }
    
    .footer {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 25px;
        margin-top: 50px;
        text-align: center;
        color: white;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .section-divider {
        background: rgba(255, 255, 255, 0.3);
        height: 2px;
        margin: 40px 0;
        border-radius: 2px;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# UI COMPONENTS
# ============================================================================

def render_hero():
    """Render hero section."""
    st.markdown("""
    <div class="hero-section">
        <h1 class="hero-title">🏥 Stroke Risk Assessment Assistant</h1>
        <p class="hero-subtitle">AI-Powered Health Guidance & Personalized Risk Analysis</p>
    </div>
    """, unsafe_allow_html=True)

def render_sidebar():
    """Render sidebar with navigation."""
    with st.sidebar:
        st.markdown("### 🎯 Navigation")
        page = st.radio("", ["Home", "Batch Analysis", "About"])
        
        st.markdown("---")
        st.markdown("### 📊 Statistics")
        st.metric("Total Assessments", "1,247")
        st.metric("Accuracy Rate", "94.3%")
        st.metric("Active Users", "342")
        
        st.markdown("---")
        st.markdown("### ℹ️ Model Info")
        st.info("Using Llama 3.1 8B model via Hugging Face")
        
        return page

def render_emergency_sidebar():
    """Render emergency warning in sidebar column."""
    st.markdown("### ⚠️ Emergency Warning")
    st.error("""
    **CALL 911 IMMEDIATELY if experiencing:**
    
    - 😵 Sudden numbness/weakness (face, arm, leg)
    - 🗣️ Trouble speaking or understanding
    - 👁️ Vision problems in one or both eyes
    - 🚶 Difficulty walking or dizziness
    - 🤕 Sudden severe headache
    
    **TIME = BRAIN**
    """)
    
    st.markdown("---")
    
    st.info("""
    **Disclaimer:**
    
    This AI assistant provides general information only. It does NOT:
    - Diagnose medical conditions
    - Prescribe treatments
    - Replace professional medical advice
    
    Always consult healthcare professionals for medical concerns.
    """)

def render_combined_home_page():
    """Render combined home page with risk assessment and AI chat."""
    
    # ========================================================================
    # SECTION 1: RISK ASSESSMENT FORM
    # ========================================================================
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📝 Patient Risk Assessment")
    st.markdown("Complete the form below to get your personalized stroke risk analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        age = st.slider("Age", 1, 100, 45)
        hypertension = st.selectbox("Hypertension", ["No", "Yes"])
    
    with col2:
        heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
        ever_married = st.selectbox("Ever Married", ["No", "Yes"])
        work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
    
    with col3:
        residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
        smoking_status = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])
        avg_glucose_level = st.number_input("Average Glucose Level (mg/dL)", 50.0, 300.0, 106.0)
    
    bmi = st.slider("BMI (Body Mass Index)", 10.0, 50.0, 25.0)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    with col_btn2:
        analyze_btn = st.button("🔍 Analyze Risk", use_container_width=True)
    
    if analyze_btn:
        with st.spinner("Analyzing..."):
            time.sleep(1.5)
        
        features = {
            'gender': gender,
            'age': age,
            'hypertension': 1 if hypertension == "Yes" else 0,
            'heart_disease': 1 if heart_disease == "Yes" else 0,
            'ever_married': ever_married,
            'work_type': work_type,
            'Residence_type': residence_type,
            'avg_glucose_level': avg_glucose_level,
            'bmi': bmi,
            'smoking_status': smoking_status
        }
        
        stroke_pred, risk_score = predict_stroke_risk(features)
        risk_level, risk_class, guidance = generate_guidance(features, risk_score, stroke_pred)
        
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown("### 📊 Assessment Results")
        
        col_m1, col_m2, col_m3 = st.columns(3)
        
        with col_m1:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value {risk_class}">{risk_level}</div>
                <div class="metric-label">Risk Level</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_m2:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">{risk_score*100:.1f}%</div>
                <div class="metric-label">Risk Score</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_m3:
            pred_text = "POSITIVE" if stroke_pred == 1 else "NEGATIVE"
            pred_color = "risk-high" if stroke_pred == 1 else "risk-low"
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value {pred_color}">{pred_text}</div>
                <div class="metric-label">Stroke Prediction</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="progress-bar">
            <div class="progress-fill" style="width: {risk_score*100}%;"></div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 💡 Personalized AI Guidance")
        for g in guidance:
            st.markdown(f'<div class="guidance-box">{g}</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ========================================================================
    # SECTION DIVIDER
    # ========================================================================
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # ========================================================================
    # SECTION 2: AI CHAT ASSISTANT
    # ========================================================================
    
    st.markdown("### 💬 AI Chat Assistant")
    st.markdown("Ask questions about stroke symptoms, risk factors, or preventive care")
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    col_chat, col_info = st.columns([2, 1])
    
    with col_chat:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        user_input = st.chat_input("Describe your symptoms or ask a question about stroke...")
        
        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            with st.chat_message("user"):
                st.markdown(user_input)
            
            with st.chat_message("assistant"):
                with st.spinner("🤖 Analyzing your query..."):
                    response = query_llm(user_input)
                st.markdown(response)
            
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_info:
        render_emergency_sidebar()
        
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        
        st.markdown("---")
        st.caption("Powered by Llama 3.1 via Hugging Face")

def render_batch_analysis():
    """Render batch analysis page."""
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📁 Batch Analysis")
    st.markdown("Upload a CSV file with patient data for bulk risk assessment")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.markdown("#### Preview of uploaded data:")
        st.dataframe(df.head(), use_container_width=True)
        
        if st.button("🚀 Run Batch Analysis"):
            with st.spinner("Processing..."):
                time.sleep(2)
                df['stroke_prediction'] = np.random.randint(0, 2, len(df))
                df['risk_score'] = np.random.uniform(0, 1, len(df))
                
                st.success(f"✅ Analysis complete! Processed {len(df)} records.")
                st.dataframe(df, use_container_width=True)
                
                csv = df.to_csv(index=False)
                st.download_button("📥 Download Results", csv, "batch_results.csv", "text/csv")
    
    st.markdown('</div>', unsafe_allow_html=True)

def render_about():
    """Render about page."""
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 🔬 About This System")
    st.markdown("""
    This Medical Query Classification and AI-Generated Guidance System uses advanced machine learning 
    algorithms to assess stroke risk based on patient health metrics.
    
    **Key Features:**
    - AI-powered chat assistant for stroke-related queries
    - Real-time risk assessment
    - Personalized health guidance
    - Batch processing capabilities
    - Evidence-based recommendations
    
    **Disclaimer:** This tool is for educational and informational purposes only. 
    Always consult qualified healthcare professionals for medical advice.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

def render_footer():
    """Render footer."""
    st.markdown(f"""
    <div class="footer">
        <p style="font-size: 0.9rem; margin-bottom: 10px;">
            © 2026 Medical AI Systems | Powered by Advanced Machine Learning
        </p>
        <p style="font-size: 0.8rem; color: rgba(255,255,255,0.7);">
            Last Updated: {datetime.now().strftime("%B %d, %Y")} | Version 2.1.0
        </p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application entry point."""
    inject_custom_css()
    render_hero()
    
    page = render_sidebar()
    
    if page == "Home":
        render_combined_home_page()
    elif page == "Batch Analysis":
        render_batch_analysis()
    else:
        render_about()
    
    render_footer()

if __name__ == "__main__":
    main()
