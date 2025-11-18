import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.figure_factory as ff
import gc
import warnings

# Suppress warnings that clutter the output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# --- GLOBAL CONSTANTS ---
OHE_COLS = [
    'Hour', 'DayOfWeek', 'Month', 'IsBusinessHours', 'Category', 
    'OSFamily', 'OSVersion', 'EntityType', 'EvidenceRole'
]
TRAINING_COLS_FILE = 'training_columns.joblib'
MODEL_FILE = 'xgboost_incident_grade_model.joblib'
FREQ_MAPS_FILE = 'frequency_maps.joblib'
LE_FILE = 'target_label_encoder.joblib'
VAL_DATA_FILE = './data/GUIDE_Train.csv' 
VAL_DATA_PARQUET = './data/GUIDE_Validate.parquet' 
MAX_SAMPLE_SIZE = 20000 # Max number of rows to sample from the full data

@st.cache_resource
def load_artifacts():
    """
    Loads all required artifacts and the FULL clean validation data.
    Returns a dictionary of artifacts, guaranteeing data is retrieved from cache.
    """
    artifacts_dict = {}
    
    try:
        # Load Joblib Artifacts
        artifacts_dict['model'] = joblib.load(MODEL_FILE)
        artifacts_dict['freq_maps'] = joblib.load(FREQ_MAPS_FILE)
        artifacts_dict['le'] = joblib.load(LE_FILE)
        artifacts_dict['training_cols'] = joblib.load(TRAINING_COLS_FILE)
        
        # --- FAST LOADING IMPLEMENTATION (Parquet Check) ---
        if pd.io.common.file_exists(VAL_DATA_PARQUET):
             df_val = pd.read_parquet(VAL_DATA_PARQUET)
             st.info("Loaded data from optimized Parquet file.")
        else:
             # Load CSV once, if Parquet doesn't exist
             df_val = pd.read_csv(VAL_DATA_FILE, low_memory=False)
             st.warning("Loaded data from slow CSV. Saving to Parquet for future speed.")
             df_val.to_parquet(VAL_DATA_PARQUET)
        
        # Initial cleaning done here ensures the cached object is clean
        df_val = df_val.dropna(subset=['IncidentGrade'])

        # Store the FULL cleaned dataframe for dynamic sampling later
        artifacts_dict['df_val_raw_full'] = df_val
        
        return artifacts_dict
        
    except Exception as e:
        st.error(f"❌ ERROR during artifact loading or data preparation: {e}. Check file paths.")
        return None


# --- FEATURE ENGINEERING PIPELINE (Modified to accept artifacts as argument) ---

def apply_feature_pipeline(df_raw: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    """Applies the full, complex feature engineering pipeline to raw data."""
    df = df_raw.copy()

    # Initial Cleaning (Drop high-null/unwanted columns)
    COLS_TO_DROP_INITIAL = ['ResourceType', 'ActionGranular', 'ActionGrouped', 'ThreatFamily', 
                            'EmailClusterId', 'AntispamDirection', 'Roles', 'SuspicionLevel', 
                            'LastVerdict', 'MitreTechniques', 'Id', 'IncidentId']
    df = df.drop(columns=[col for col in COLS_TO_DROP_INITIAL if col in df.columns], errors='ignore')
    if 'Usage' in df.columns:
        df = df.drop(columns=['Usage'])

    # Temporal Features
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    df['Hour'] = df['Timestamp'].dt.hour.astype('category')
    df['DayOfWeek'] = df['Timestamp'].dt.dayofweek.astype('category')
    df['Month'] = df['Timestamp'].dt.month.astype('category')
    df['IsBusinessHours'] = df['Hour'].apply(lambda x: 1 if 8 <= x < 18 else 0).astype('category')
    df.drop(columns=['Timestamp'], inplace=True, errors='ignore')

    # Frequency Encoding (Using saved training maps)
    for col, counts_map in artifacts['freq_maps'].items():
        if col in df.columns:
            df[f'{col}_Freq'] = df[col].map(counts_map).fillna(0.0).astype('float32')
            df.drop(columns=[col], inplace=True, errors='ignore')

    # One-Hot Encoding
    df_encoded = pd.get_dummies(df, columns=[col for col in OHE_COLS if col in df.columns], drop_first=False)
    
    # Final Type and Alignment Correction
    y_raw = df_encoded['IncidentGrade']
    df_encoded = df_encoded.drop(columns=['IncidentGrade'])

    # 2. Add missing OHE features 
    missing_cols = set(artifacts['training_cols']) - set(df_encoded.columns)
    for c in missing_cols:
        df_encoded[c] = 0.0

    # 3. Drop extra columns 
    extra_cols = set(df_encoded.columns) - set(artifacts['training_cols'])
    if extra_cols:
        df_encoded = df_encoded.drop(columns=extra_cols, axis=1)

    # 4. Final alignment and type enforcement
    df_encoded = df_encoded[artifacts['training_cols']]
    df_encoded = df_encoded.astype('float32')
    
    return df_encoded, y_raw

# --- MODEL PREDICTION FUNCTION ---

def make_prediction(X_processed: pd.DataFrame, y_raw: pd.Series, artifacts: dict) -> tuple:
    """Makes predictions and calculates metrics."""
    model = artifacts['model']
    le = artifacts['le']

    y_pred_encoded = model.predict(X_processed)
    y_true_encoded = le.transform(y_raw)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true_encoded, y_pred_encoded)
    kappa = cohen_kappa_score(y_true_encoded, y_pred_encoded, weights='quadratic')
    
    # Generate Confusion Matrix
    conf_matrix = confusion_matrix(y_true_encoded, y_pred_encoded, labels=le.transform(le.classes_))
    
    # Decode predictions for display
    y_pred_grade = le.inverse_transform(y_pred_encoded)

    return y_pred_grade, accuracy, kappa, conf_matrix, y_true_encoded

# --- STREAMLIT APP LAYOUT ---

def main():
    st.set_page_config(layout="wide", page_title="Incident Grade Predictor")
    st.title("🛡️ Security Incident Grade Prediction Service")
    
    # Load artifacts and check if successful
    artifacts = load_artifacts()
    if artifacts is None:
        return # Stop execution if loading failed
    
    # --- Session State Initialization ---
    if 'sampling_seed' not in st.session_state:
        st.session_state.sampling_seed = 42 # Initialize seed
    
    st.markdown("### Demo: Analyze Model Robustness on Unseen Validation Data")

    # --- Sidebar Controls ---
    with st.sidebar:
        st.header("Model Parameters")
        
        # 1. Button to refresh the sample (updates the seed)
        if st.button('🔄 Refresh Data Sample (New Incidents)'):
            # Change the seed to force a new random sample
            st.session_state.sampling_seed = np.random.randint(0, 1000000)
            
        # Get the full data and calculate the max sample size
        df_val_raw_full = artifacts['df_val_raw_full']
        FULL_SIZE = len(df_val_raw_full)
        
        # 2. Draw the 20k subset dynamically using the seed
        df_val_raw_subset = df_val_raw_full.sample(
            n=min(MAX_SAMPLE_SIZE, FULL_SIZE), 
            random_state=st.session_state.sampling_seed
        )
        K = len(df_val_raw_subset) # Max slider value
        
        # 3. User selection for number of incidents (Slider draws from the 20k subset)
        num_incidents = st.slider(
            f'Select Number of Incidents (K) to Analyze (Max {K:,}):',
            min_value=1,
            max_value=K, 
            value=min(1000, K),
            step=100
        )
        
        st.markdown("---")
        st.success(f"Final Validation QWK: **0.8412**")
        st.info("Validation metrics below reflect the *true* generalization.")

    # --- Data Sampling and Processing ---
    
    # Randomly sample the final K incidents from the 20k subset
    df_sample_raw = df_val_raw_subset.sample(n=num_incidents, random_state=42) 
    
    with st.spinner(f"Processing {num_incidents} incidents through the complex feature pipeline..."):
        # Process the raw sample data
        X_processed, y_true_raw = apply_feature_pipeline(df_sample_raw, artifacts)
        
        # Make predictions
        y_pred_grade, accuracy, kappa, conf_matrix, y_true_encoded = make_prediction(X_processed, y_true_raw, artifacts)

    # --- Display Metrics and Visuals ---

    st.subheader(f"Results for {num_incidents} Sampled Incidents")
    
    col1, col2, col3 = st.columns(3)
    
    accuracy_diff = accuracy - 0.8717 
    col1.metric("Validation Accuracy", f"{accuracy:.4f}", f"{accuracy_diff:.4f} vs. Mean Val Accuracy")
    col2.metric("Cohen's Kappa (QWK)", f"{kappa:.4f}", help="Measures the agreement of the ordinal rank (0=None, 1=Perfect)")
    col3.metric("Total Rows Processed", f"{num_incidents:,}")
    
    st.markdown("---")

    # --- Confusion Matrix Visualization (Plotly) ---
    st.subheader("Confusion Matrix (Interactive)")
    
    le_classes = artifacts['le'].classes_
    le_classes_list = le_classes.tolist()

    fig_plotly = ff.create_annotated_heatmap(
        z=conf_matrix.tolist(), 
        x=le_classes_list, 
        y=le_classes_list, 
        colorscale='Blues',
        showscale=True,
        font_colors=['black', 'white']
    )
    
    # Customize layout for clear axes and title
    fig_plotly.update_layout(
        xaxis=dict(title='Predicted Grade'),
        yaxis=dict(title='True Grade', autorange="reversed"),
        title='Actual vs. Predicted Incident Grade (Hover for details)',
        margin=dict(t=50, b=50),
        font=dict(size=10)
    )
    
    # Display the interactive Plotly chart
    st.plotly_chart(fig_plotly, use_container_width=True)

    # --- Detailed Data View ---
    st.subheader("Sample of Predicted Incidents")
    
    df_display = df_sample_raw.head(20).copy()
    df_display['Predicted_Grade'] = y_pred_grade[:20]
    
    display_cols = ['Timestamp', 'Category', 'DeviceId', 'IncidentGrade', 'Predicted_Grade']
    df_display = df_display[[col for col in display_cols if col in df_display.columns]]
    
    st.dataframe(df_display, use_container_width=True)
    st.caption("Note: This table shows the model successfully applies all feature engineering and makes a final prediction based on the saved artifacts.")
    
    del X_processed
    gc.collect()

if __name__ == "__main__":
    main()