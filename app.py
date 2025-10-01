#for english users
import streamlit as st
import sys

# === DEBUG CODE - ADD THIS FIRST ===
st.write("Python version:", sys.version)
st.write("Python path:", sys.path)

try:
    import torch
    st.success("Torch imported successfully!")
    st.write("Torch version:", torch.__version__)
except ImportError as e:
    st.error(f"Cannot import torch: {e}")
    
    # Check what packages ARE installed
    import subprocess
    result = subprocess.run(['pip', 'list'], capture_output=True, text=True)
    st.text(result.stdout)
    st.stop()  # Stop here if torch fails

#for english users
import streamlit as st
import torch
import numpy as np
import pandas as pd
import pickle
import scipy.special
import warnings
import plotly.graph_objects as go
import plotly.express as px
warnings.filterwarnings('ignore')


# === SET PAGE CONFIG - MUST BE FIRST STREAMLIT COMMAND ===
st.set_page_config(
    page_title="IVFEngine: A deep learning based live baby birth prediction system",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# === PASSWORD PROTECTION ===
def check_password():
    """Returns `True` if the user had the correct password."""
    
    # PASSWORD FOR REVIEWERS/EDITORS
    correct_password = "review123"
    
    # Initialize session state
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False
    
    # If already authenticated, show the app
    if st.session_state["password_correct"]:
        return True
    
    # Show password input form
    st.markdown("""
    <div style='text-align: center; padding: 2rem;'>
        <h1>🔒 IVFEngine: A deep learning based live baby birth prediction system</h1>
        <p><strong>The manuscript is currently under submission stages so access is restricted to journal reviewers and editors only.</strong></p>
        <p style='color: #666; font-size: 0.9em;'>
    </div>
    """, unsafe_allow_html=True)
    
    # Create columns for centered layout
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        password_input = st.text_input(
            "Enter access password:", 
            type="password", 
            key="password_input"
        )
        
        submit_button = st.button("Submit Password")
        
        if submit_button:
            if password_input == correct_password:
                st.session_state["password_correct"] = True
                st.rerun()
            else:
                st.error("❌ Incorrect password. Please contact the corresponding author for access.")
    
    return False

# Check password before showing the app
if not check_password():
    st.stop()




# === YOUR ORIGINAL APP CODE STARTS HERE ===
# IMPORTANT: REMOVE ANY OTHER st.set_page_config() CALLS FROM YOUR ORIGINAL CODE

# Import the FTTransformer model class
try:
    from rtdl_revisiting_models import FTTransformer
except ImportError:
    st.error("rtdl_revisiting_models module not found. Please ensure it's installed and available.")
    st.stop()

# Advanced CSS for journal-quality styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Source+Serif+Pro:wght@400;600&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    .main-container {
        max-width: 1400px;
        margin: 0 auto;
        padding: 2rem;
    }
    
    .journal-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        box-shadow: 0 20px 40px rgba(102, 126, 234, 0.3);
        margin-bottom: 3rem;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    
    .journal-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: url("data:image/svg+xml,%3Csvg width='40' height='40' viewBox='0 0 40 40' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='%23ffffff' fill-opacity='0.05' fill-rule='evenodd'%3E%3Cpath d='m0 40l40-40h-40v40zm40 40v-40h-40l40 40z'/%3E%3C/g%3E%3C/svg%3E");
        animation: float 20s infinite linear;
    }
    
    @keyframes float {
        0% { transform: translate(-50%, -50%) rotate(0deg); }
        100% { transform: translate(-50%, -50%) rotate(360deg); }
    }
    
    .journal-title {
        font-family: 'Source Serif Pro', serif;
        font-size: 2.5rem;
        font-weight: 600;
        color: white;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        position: relative;
        z-index: 2;
    }
    
    .journal-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem;
        font-weight: 400;
        color: rgba(255,255,255,0.9);
        margin-bottom: 1rem;
        position: relative;
        z-index: 2;
    }
    
    .journal-badge {
        display: inline-block;
        background: rgba(255,255,255,0.2);
        backdrop-filter: blur(10px);
        padding: 0.5rem 1.5rem;
        border-radius: 50px;
        font-family: 'Inter', sans-serif;
        font-size: 0.9rem;
        font-weight: 500;
        color: white;
        border: 1px solid rgba(255,255,255,0.3);
        position: relative;
        z-index: 2;
    }
    
    .clinical-form {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 25px 50px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.8);
        backdrop-filter: blur(20px);
        margin-bottom: 2rem;
        position: relative;
        height: fit-content;
    }
    
    .clinical-form::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #667eea, #764ba2);
        border-radius: 20px 20px 0 0;
    }
    
    .form-section-title {
        font-family: 'Source Serif Pro', serif;
        font-size: 1.8rem;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 2rem;
        text-align: center;
        position: relative;
    }
    
    .form-section-title::after {
        content: '';
        position: absolute;
        bottom: -10px;
        left: 50%;
        transform: translateX(-50%);
        width: 60px;
        height: 3px;
        background: linear-gradient(90deg, #667eea, #764ba2);
        border-radius: 2px;
    }
    
    .stSelectbox > div > div {
        background: white;
        border: 2px solid #e2e8f0;
        border-radius: 8px;
        font-family: 'Inter', sans-serif;
        font-weight: 400;
        transition: all 0.3s ease;
        margin-bottom: 1rem;
    }
    
    /* Match text size for dropdown options */
    .stSelectbox div[data-baseweb="select"] > div {
        font-family: 'Inter', sans-serif;
        font-size: 0.95rem;
        font-weight: 400;
    }
    
    .stSelectbox > div > div:focus-within {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    .stSelectbox label {
        font-family: 'Inter', sans-serif;
        font-size: 0.95rem;
        font-weight: 600;
        color: #34495e;
        margin-bottom: 0.5rem;
    }
    
    .results-container {
        background: white;
        border-radius: 20px;
        padding: 3rem;
        box-shadow: 0 25px 50px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.8);
        text-align: center;
        position: relative;
        overflow: hidden;
        margin-bottom: 2rem;
    }
    
    .results-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        border-radius: 20px 20px 0 0;
    }
    
    .results-container.high-prob::before {
        background: linear-gradient(90deg, #e74c3c, #c0392b);
    }
    
    .results-container.medium-prob::before {
        background: linear-gradient(90deg, #27ae60, #229954);
    }
    
    .results-container.low-prob::before {
        background: linear-gradient(90deg, #f39c12, #e67e22);
    }
    
    .probability-score {
        font-family: 'Source Serif Pro', serif;
        font-size: 4rem;
        font-weight: 700;
        margin: 1rem 0;
        line-height: 1;
    }
    
    .probability-label {
        font-family: 'Inter', sans-serif;
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 1rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .waterfall-container {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 25px 50px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.8);
        margin-bottom: 2rem;
        position: relative;
    }
    
    .waterfall-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #667eea, #764ba2);
        border-radius: 20px 20px 0 0;
    }
    
    .methodology-note {
        background: #f1f5f9;
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 2rem;
        font-family: 'Inter', sans-serif;
        font-size: 0.9rem;
        color: #64748b;
        border-left: 4px solid #94a3b8;
    }
    
    .disclaimer {
        text-align: center;
        font-family: 'Inter', sans-serif;
        font-size: 0.85rem;
        color: #64748b;
        margin-top: 2rem;
        font-style: italic;
    }
    
    /* Hide Streamlit branding */
    .stApp > header {visibility: hidden;}
    .stApp > div.stMainBlockContainer {padding-top: 1rem;}
    
    /* Custom button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 1rem 2rem !important;
        color: white !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        width: 100% !important;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3) !important;
        transition: all 0.3s ease !important;
        margin-top: 2rem !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.4) !important;
    }
    
    /* Remove extra spacing */
    .element-container {
        margin-bottom: 0.5rem !important;
    }
    
    .stSelectbox {
        margin-bottom: 1rem !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_preprocessor():
    """Load the trained model and preprocessor"""
    try:
        # Load model
        model_checkpoint = torch.load('complete_model.pth', map_location='cpu')
        
        # Extract model configuration
        config = model_checkpoint['model_config']
        feature_names = model_checkpoint['feature_names']
        
        # Initialize model
        model = FTTransformer(
            n_cont_features=config['n_cont_features'],
            cat_cardinalities=config['cat_cardinalities'],
            d_out=config['d_out'],
            **FTTransformer.get_default_kwargs(),
        )
        
        # Load model weights
        model.load_state_dict(model_checkpoint['model_state_dict'])
        model.eval()
        
        # Load preprocessor
        with open('preprocessor.pkl', 'rb') as f:
            preprocessor = pickle.load(f)
        
        return model, preprocessor, model_checkpoint, feature_names
    
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.error("Please ensure 'complete_model.pth' and 'preprocessor.pkl' are in the same directory as this script.")
        return None, None, None, None

def create_feature_dataframe(input_data):
    """Create a dataframe exactly as it appears in training"""
    df = pd.DataFrame([input_data])
    
    for col in df.columns:
        if col in ['type of infertility - female secondary', 'cause of infertility - female factors', 
                   'type of infertility - male primary', 'cause of infertility - partner sperm concentration',
                   'type of infertility - male secondary', 'cause of infertility - partner sperm motility']:
            df[col] = df[col].astype('object')
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def preprocess_input(input_data, preprocessor, feature_names, model_checkpoint):
    """Preprocess user input data exactly as in training"""
    try:
        df = create_feature_dataframe(input_data)
        config = model_checkpoint['model_config']
        
        catfeatures = df.select_dtypes(include=['object']).columns.tolist()
        numfeatures = df.drop(catfeatures, axis=1).columns.tolist()
        
        if numfeatures:
            X_cont = df[numfeatures].to_numpy().astype(np.float32)
            X_cont_processed = preprocessor.transform(X_cont)
        else:
            X_cont_processed = np.array([]).reshape(1, 0)
        
        if catfeatures:
            X_cat = np.zeros((1, len(catfeatures)), dtype=np.int64)
            for i, col in enumerate(catfeatures):
                X_cat[0, i] = int(df[col].iloc[0])
        else:
            X_cat = None
        
        return X_cont_processed, X_cat
        
    except Exception as e:
        st.error(f"Error preprocessing input: {str(e)}")
        return None, None

def make_prediction(model, X_cont, X_cat):
    """Make prediction using the trained model"""
    try:
        with torch.no_grad():
            x_cont_tensor = torch.tensor(X_cont, dtype=torch.float32) if X_cont is not None else None
            x_cat_tensor = torch.tensor(X_cat, dtype=torch.int64) if X_cat is not None else None
            
            prediction = model(x_cont_tensor, x_cat_tensor)
            probability = scipy.special.expit(prediction.numpy()[0][0])
            
            return probability
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None

def get_dynamic_baseline():
    """Create a dynamic baseline based on typical/neutral values"""
    return {
        'embryos transfered': 1,
        'type of infertility - female secondary': 0,
        'cause of infertility - female factors': 0,
        'type of infertility - male primary': 1,
        'type of infertility - male secondary': 0,
        'cause of infertility - partner sperm concentration': 0,
        'cause of infertility - partner sperm motility': 0
    }

def calculate_feature_contributions(input_data, model, preprocessor, model_info, feature_names):
    """Calculate feature contributions starting from a neutral baseline"""
    try:
        baseline_data = get_dynamic_baseline()
        
        # Get baseline prediction (this is just for reference, not for the chart)
        X_cont_baseline, X_cat_baseline = preprocess_input(baseline_data, preprocessor, feature_names, model_info)
        baseline_prob = make_prediction(model, X_cont_baseline, X_cat_baseline)
        
        # Get actual prediction
        X_cont_actual, X_cat_actual = preprocess_input(input_data, preprocessor, feature_names, model_info)
        actual_prob = make_prediction(model, X_cont_actual, X_cat_actual)
        
        feature_mapping = {
            'embryos transfered': 'Embryos Transferred',
            'type of infertility - female secondary': 'Female Secondary Infertility',
            'cause of infertility - female factors': 'Female Factors',
            'type of infertility - male primary': 'Male Primary Infertility',
            'type of infertility - male secondary': 'Male Secondary Infertility',
            'cause of infertility - partner sperm concentration': 'Sperm Concentration',
            'cause of infertility - partner sperm motility': 'Sperm Motility'
        }
        
        feature_contributions = {}
        
        for feature_key, display_name in feature_mapping.items():
            test_data = baseline_data.copy()
            test_data[feature_key] = input_data[feature_key]
            
            X_cont_test, X_cat_test = preprocess_input(test_data, preprocessor, feature_names, model_info)
            test_prob = make_prediction(model, X_cont_test, X_cat_test)
            
            contribution = test_prob - baseline_prob
            feature_contributions[display_name] = contribution
        
        return feature_contributions, baseline_prob, actual_prob
        
    except Exception as e:
        st.error(f"Error calculating feature contributions: {str(e)}")
        return {}, 0, 0

def create_waterfall_plot(feature_contributions, actual_prob):
    """Create a clean waterfall plot showing only feature contributions and final prediction"""
    try:
        # Prepare data - sort features by absolute contribution
        features = list(feature_contributions.keys())
        contributions = list(feature_contributions.values())
        
        sorted_indices = sorted(range(len(contributions)), key=lambda i: abs(contributions[i]), reverse=True)
        features = [features[i] for i in sorted_indices]
        contributions = [contributions[i] for i in sorted_indices]
        
        # Create the plot data - only feature contributions and final result
        x_labels = features + ['Final Prediction']
        y_values = contributions + [0]  # Final bar height is 0 (it's a total bar)
        
        # Use Plotly's waterfall with proper measure types
        measure = ['relative'] * len(features) + ['total']
        
        fig = go.Figure(go.Waterfall(
            name=" ",
            orientation="v",
            measure=measure,
            x=x_labels,
            y=y_values,
            text=[f"{c:+.3f}" for c in contributions] + [f"{actual_prob:.3f}"],
            textposition="outside",
            connector={"line":{"color":"rgb(63, 63, 63)"}},
            decreasing={"marker":{"color":"#e74c3c"}},
            increasing={"marker":{"color":"#27ae60"}},
            totals={"marker":{"color":"#667eea"}}
        ))
        
        # Update layout for better readability
        fig.update_layout(
            title={
                'text': 'The waterfall plot for feature contribution',
                'x': 0.5,
                'font': {'size': 18, 'family': 'Source Serif Pro, serif'}
            },
            xaxis_title='Clinical Features',
            yaxis_title='Probability Impact',
            height=500,
            template='plotly_white',
            font={'family': 'Inter, sans-serif'},
            showlegend=False,
            margin=dict(l=50, r=50, t=80, b=100),
            xaxis={'tickangle': -45, 'tickfont': {'size': 10}},
            yaxis={'range': [min(contributions + [0]) - 0.1, max([sum(contributions), 0]) + 0.1]}
        )
        
        # Add reference line at y=0
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating waterfall plot: {str(e)}")
        return None

def main():
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    # Journal-style header - removed the badge
    st.markdown("""
    <div class="journal-header">
        <h1 class="journal-title">IVFEngine: A deep learning based live baby birth prediction system</h1>
        <p class="journal-subtitle">Advanced Clinical Decision Support using Deep Learning</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model and preprocessor
    model, preprocessor, model_info, feature_names = load_model_and_preprocessor()
    
    if model is None:
        st.stop()
    
    # Create two columns for better layout
    col1, col2 = st.columns([1, 1.2], gap="large")
    
    with col1:
        st.markdown("""
        <div class="clinical-form">
            <h2 class="form-section-title">Patient Parameters</h2>
        """, unsafe_allow_html=True)
        
        feature_inputs = {}
        
        embryos_transferred = st.selectbox(
            "Embryos transferred",
            options=[1, 2, 0, 3],
            index=0,
            help="Number of embryos transferred during the IVF cycle"
        )
        feature_inputs['embryos transfered'] = embryos_transferred
        
        # Modified: Female secondary - Changed labels
        female_secondary_options = ["Unknown", "No", "Yes"]
        female_secondary_values = [0, 1, 2]
        female_secondary_display = st.selectbox(
            "Type of infertility - Female secondary",
            options=female_secondary_options,
            index=0,
            help="Secondary infertility classification for female partner"
        )
        feature_inputs['type of infertility - female secondary'] = female_secondary_values[female_secondary_options.index(female_secondary_display)]
        
        # Modified: Female factors - Changed labels
        female_factors_options = ["No", "Yes"]
        female_factors_values = [0, 1]
        female_factors_display = st.selectbox(
            "Cause of infertility - Female factors",
            options=female_factors_options,
            index=0,
            help="Primary female factors contributing to infertility"
        )
        feature_inputs['cause of infertility - female factors'] = female_factors_values[female_factors_options.index(female_factors_display)]
        
        # Modified: Male primary - Changed labels and removed option 2
        male_primary_options = ["No", "Yes"]
        male_primary_values = [0, 1]
        male_primary_display = st.selectbox(
            "Type of infertility - Male primary",
            options=male_primary_options,
            index=1,  # Default to "Yes" (value 1)
            help="Primary male infertility classification"
        )
        feature_inputs['type of infertility - male primary'] = male_primary_values[male_primary_options.index(male_primary_display)]
        
        # Modified: Male secondary - Changed labels
        male_secondary_options = ["No", "Yes"]
        male_secondary_values = [0, 1]
        male_secondary_display = st.selectbox(
            "Type of infertility - Male secondary",
            options=male_secondary_options,
            index=0,
            help="Secondary male infertility classification"
        )
        feature_inputs['type of infertility - male secondary'] = male_secondary_values[male_secondary_options.index(male_secondary_display)]
        
        # Modified: Sperm concentration - Changed labels
        sperm_concentration_options = ["Unknown", "No", "Yes"]
        sperm_concentration_values = [0, 1, 2]
        sperm_concentration_display = st.selectbox(
            "Cause of infertility - Partner sperm concentration",
            options=sperm_concentration_options,
            index=0,
            help="Sperm concentration issues: Unknown (normal), No (mild issues), Yes (severe issues)"
        )
        feature_inputs['cause of infertility - partner sperm concentration'] = sperm_concentration_values[sperm_concentration_options.index(sperm_concentration_display)]
        
        # Modified: Sperm motility - Changed labels
        sperm_motility_options = ["Unknown", "No", "Yes"]
        sperm_motility_values = [0, 1, 2]
        sperm_motility_display = st.selectbox(
            "Cause of infertility - Partner sperm motility",
            options=sperm_motility_options,
            index=0,
            help="Sperm motility issues: Unknown (normal), No (mild issues), Yes (severe issues)"
        )
        feature_inputs['cause of infertility - partner sperm motility'] = sperm_motility_values[sperm_motility_options.index(sperm_motility_display)]
        
        predict_button = st.button("🔬 Predict", type="primary", use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        if predict_button:
            X_cont, X_cat = preprocess_input(feature_inputs, preprocessor, feature_names, model_info)
            
            if X_cont is not None:
                probability = make_prediction(model, X_cont, X_cat)
                
                if probability is not None:
                    st.session_state['prediction'] = probability
                    st.session_state['input_data'] = feature_inputs.copy()
        
        if 'prediction' in st.session_state:
            probability = st.session_state['prediction']
            input_data = st.session_state['input_data']
            
            if probability >= 0.70:
                prob_class = "high-prob"
                text_color = "#e74c3c"
                category = "High Probability"
            elif probability >= 0.50:
                prob_class = "medium-prob"
                text_color = "#27ae60"
                category = "Medium Probability"
            else:
                prob_class = "low-prob"
                text_color = "#f39c12"
                category = "Low Probability"
            
            st.markdown(f"""
            <div class="results-container {prob_class}">
                <div class="probability-label" style="color: {text_color};">{category}</div>
                <div class="probability-score" style="color: {text_color};">{probability:.3f}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Waterfall plot section
            st.markdown("""
            <div class="waterfall-container">
            """, unsafe_allow_html=True)
            
            feature_contributions, baseline_prob, actual_prob = calculate_feature_contributions(
                input_data, model, preprocessor, model_info, feature_names
            )
            
            # Create and display the improved waterfall plot
            waterfall_fig = create_waterfall_plot(feature_contributions, actual_prob)
            
            if waterfall_fig:
                st.plotly_chart(waterfall_fig, use_container_width=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Methodology and disclaimer
    st.markdown("""
    <div class="methodology-note">
        <strong>Methodology:</strong> The waterfall chart shows how each clinical feature contributes to your final prediction. Positive values (green) increase the probability, while negative values (red) decrease it. The chart starts from a neutral baseline and accumulates feature impacts to reach your final prediction.
    </div>
    
    <div class="disclaimer">
        <strong>Clinical Disclaimer:</strong> This tool is intended for research and educational purposes only. All clinical decisions should be made in consultation with qualified healthcare professionals.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":

    main()
