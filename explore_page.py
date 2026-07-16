import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
def initialize_session_state():
    """Initialize session state variables"""
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'feature_importance' not in st.session_state:
        st.session_state.feature_importance = None
    if 'pca_result' not in st.session_state:
        st.session_state.pca_result = None
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False

initialize_session_state()

# ============================================================================
# DATA LOADING AND CACHING
# ============================================================================
@st.cache_data
def load_data():
    """Load and preprocess the heart disease dataset"""
    try:
        df = pd.read_csv('heart.csv')
        
        # Data preprocessing
        # Rename columns for better readability
        column_mapping = {
            'age': 'Age',
            'sex': 'Sex',
            'cp': 'Chest Pain Type',
            'trestbps': 'Resting Blood Pressure',
            'chol': 'Cholesterol',
            'fbs': 'Fasting Blood Sugar',
            'restecg': 'Resting ECG',
            'thalach': 'Max Heart Rate',
            'exang': 'Exercise Induced Angina',
            'oldpeak': 'ST Depression',
            'slope': 'ST Slope',
            'ca': 'Major Vessels',
            'thal': 'Thalassemia',
            'target': 'Heart Disease'
        }
        df = df.rename(columns=column_mapping)
        
        return df
    except FileNotFoundError:
        st.error("❌ Dataset 'heart.csv' not found. Please ensure the file exists in the current directory.")
        return pd.DataFrame()

@st.cache_data
def get_feature_importance(df):
    """Calculate feature importance using Random Forest"""
    # Prepare data
    X = df.drop('Heart Disease', axis=1)
    y = df['Heart Disease']
    
    # Handle categorical variables
    X = pd.get_dummies(X, drop_first=True)
    
    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    # Get feature importance
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    return importance_df, rf

@st.cache_data
def perform_pca(df):
    """Perform PCA analysis on the dataset"""
    # Prepare data
    X = df.drop('Heart Disease', axis=1)
    X = pd.get_dummies(X, drop_first=True)
    
    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X_scaled)
    
    # Create PCA dataframe
    pca_df = pd.DataFrame({
        'PC1': pca_result[:, 0],
        'PC2': pca_result[:, 1],
        'Heart Disease': df['Heart Disease']
    })
    
    return pca_df, pca, X_scaled

# ============================================================================
# MAIN EXPLORE PAGE
# ============================================================================
def show_explore_page():
    st.title("🔬 Advanced Heart Disease Data Exploration")
    st.markdown("Comprehensive analysis and visualization of heart disease dataset with advanced analytics")
    
    # Load data
    df = load_data()
    if df.empty:
        return
    
    st.session_state.df = df
    
    # Dataset Overview
    st.header("📊 Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        st.metric("Total Features", len(df.columns) - 1)
    with col3:
        disease_count = df['Heart Disease'].sum()
        st.metric("Disease Cases", f"{disease_count:,}")
    with col4:
        disease_pct = (disease_count / len(df)) * 100
        st.metric("Disease Prevalence", f"{disease_pct:.1f}%")
    
    # Tabs for different analyses
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Target Analysis", 
        "🔗 Correlations", 
        "📊 Feature Distribution",
        "🧠 PCA Analysis",
        "🎯 Feature Importance",
        "📋 Data Preview"
    ])
    
    # ========================================================================
    # TAB 1: TARGET ANALYSIS
    # ========================================================================
    with tab1:
        st.subheader("🎯 Target Variable Analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Enhanced target distribution with Plotly
            target_counts = df['Heart Disease'].value_counts().reset_index()
            target_counts.columns = ['Heart Disease', 'Count']
            target_counts['Heart Disease'] = target_counts['Heart Disease'].map({0: 'No Disease', 1: 'Disease'})
            
            fig_target = px.pie(
                target_counts, 
                values='Count', 
                names='Heart Disease',
                title='Distribution of Heart Disease',
                color='Heart Disease',
                color_discrete_map={'No Disease': '#2ecc71', 'Disease': '#e74c3c'},
                hole=0.3
            )
            fig_target.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_target, use_container_width=True)
        
        with col2:
            st.markdown("### Key Insights")
            no_disease = len(df[df['Heart Disease'] == 0])
            disease = len(df[df['Heart Disease'] == 1])
            
            st.info(f"""
            **Dataset Balance:**
            - 🟢 No Disease: {no_disease} ({no_disease/len(df)*100:.1f}%)
            - 🔴 Disease: {disease} ({disease/len(df)*100:.1f}%)
            
            **Class Balance:** {'✅ Balanced' if 0.4 <= disease/len(df) <= 0.6 else '⚠️ Imbalanced'}
            """)
        
        # Target vs Features
        st.subheader("Target vs Key Features")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Age distribution by target
            fig_age = px.box(
                df, x='Heart Disease', y='Age',
                title='Age Distribution by Heart Disease',
                color='Heart Disease',
                color_discrete_map={0: '#2ecc71', 1: '#e74c3c'},
                labels={'Heart Disease': 'Heart Disease', 'Age': 'Age'}
            )
            fig_age.update_xaxis(tickvals=[0, 1], ticktext=['No Disease', 'Disease'])
            st.plotly_chart(fig_age, use_container_width=True)
        
        with col2:
            # Max Heart Rate by target
            fig_hr = px.box(
                df, x='Heart Disease', y='Max Heart Rate',
                title='Max Heart Rate by Heart Disease',
                color='Heart Disease',
                color_discrete_map={0: '#2ecc71', 1: '#e74c3c'},
                labels={'Heart Disease': 'Heart Disease', 'Max Heart Rate': 'Max Heart Rate'}
            )
            fig_hr.update_xaxis(tickvals=[0, 1], ticktext=['No Disease', 'Disease'])
            st.plotly_chart(fig_hr, use_container_width=True)
    
    # ========================================================================
    # TAB 2: CORRELATIONS
    # ========================================================================
    with tab2:
        st.subheader("🔗 Correlation Analysis")
        
        # Correlation matrix
        numeric_df = df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr()
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Interactive correlation heatmap
            fig_corr = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                title="Feature Correlation Matrix",
                color_continuous_scale="RdBu_r",
                zmin=-1, zmax=1
            )
            fig_corr.update_layout(height=600)
            st.plotly_chart(fig_corr, use_container_width=True)
        
        with col2:
            st.markdown("### Top Correlations")
            
            # Get top correlations with target
            target_corr = corr_matrix['Heart Disease'].sort_values(ascending=False)
            top_features = target_corr.head(6)
            
            for feature, corr_val in top_features.items():
                if feature != 'Heart Disease':
                    color = '🟢' if corr_val > 0 else '🔴'
                    st.metric(feature, f"{color} {corr_val:.3f}")
        
        # Correlation pairs
        with st.expander("📊 Correlation Pair Analysis"):
            st.markdown("#### Strongest Correlations")
            
            # Get top correlation pairs (excluding self-correlation)
            corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_pairs.append({
                        'Feature 1': corr_matrix.columns[i],
                        'Feature 2': corr_matrix.columns[j],
                        'Correlation': corr_matrix.iloc[i, j]
                    })
            
            corr_pairs_df = pd.DataFrame(corr_pairs)
            corr_pairs_df = corr_pairs_df.sort_values('Correlation', ascending=False)
            
            st.dataframe(corr_pairs_df.head(20), use_container_width=True)
    
    # ========================================================================
    # TAB 3: FEATURE DISTRIBUTION
    # ========================================================================
    with tab3:
        st.subheader("📊 Feature Distribution Analysis")
        
        # Select features to visualize
        features = [col for col in df.columns if col != 'Heart Disease']
        selected_features = st.multiselect(
            "Select features to visualize",
            options=features,
            default=features[:4]
        )
        
        if selected_features:
            # Create subplots
            n_features = len(selected_features)
            n_cols = 2
            n_rows = (n_features + 1) // n_cols
            
            fig = make_subplots(
                rows=n_rows, cols=n_cols,
                subplot_titles=selected_features,
                specs=[[{'type': 'violin'} for _ in range(min(n_cols, n_features - i*n_cols))] 
                       for i in range(n_rows)]
            )
            
            for idx, feature in enumerate(selected_features):
                row = idx // n_cols + 1
                col = idx % n_cols + 1
                
                # Add violin plot for each feature
                for disease_val in [0, 1]:
                    subset = df[df['Heart Disease'] == disease_val]
                    disease_label = 'No Disease' if disease_val == 0 else 'Disease'
                    color = '#2ecc71' if disease_val == 0 else '#e74c3c'
                    
                    fig.add_trace(
                        go.Violin(
                            y=subset[feature],
                            name=disease_label,
                            box_visible=True,
                            line_color=color,
                            fillcolor=color,
                            opacity=0.6,
                            legendgroup=disease_label,
                            showlegend=(idx == 0)
                        ),
                        row=row, col=col
                    )
            
            fig.update_layout(height=400*n_rows, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
    
    # ========================================================================
    # TAB 4: PCA ANALYSIS
    # ========================================================================
    with tab4:
        st.subheader("🧠 PCA Dimensionality Reduction")
        
        # Perform PCA
        pca_df, pca_model, X_scaled = perform_pca(df)
        st.session_state.pca_result = pca_df
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # PCA scatter plot
            fig_pca = px.scatter(
                pca_df,
                x='PC1',
                y='PC2',
                color='Heart Disease',
                title='PCA Visualization of Heart Disease Data',
                color_discrete_map={0: '#2ecc71', 1: '#e74c3c'},
                labels={'Heart Disease': 'Heart Disease'},
                hover_data={'Heart Disease': True}
            )
            
            # Add centroids
            for disease_val in [0, 1]:
                subset = pca_df[pca_df['Heart Disease'] == disease_val]
                centroid_x = subset['PC1'].mean()
                centroid_y = subset['PC2'].mean()
                color = '#2ecc71' if disease_val == 0 else '#e74c3c'
                label = 'No Disease' if disease_val == 0 else 'Disease'
                
                fig_pca.add_trace(
                    go.Scatter(
                        x=[centroid_x],
                        y=[centroid_y],
                        mode='markers',
                        marker=dict(size=15, color=color, symbol='x'),
                        name=f'Centroid - {label}'
                    )
                )
            
            fig_pca.update_layout(height=500)
            st.plotly_chart(fig_pca, use_container_width=True)
        
        with col2:
            st.markdown("### Explained Variance")
            
            # Explained variance
            explained_var = pca_model.explained_variance_ratio_
            
            for i, var in enumerate(explained_var[:2]):
                st.metric(f"PC{i+1}", f"{var*100:.2f}%")
            
            st.metric("Total Variance", f"{explained_var[:2].sum()*100:.2f}%")
        
        # PCA Components
        with st.expander("📊 PCA Component Analysis"):
            st.markdown("#### Component Loadings")
            
            # Get feature names
            feature_names = pd.get_dummies(df.drop('Heart Disease', axis=1), drop_first=True).columns
            
            loadings_df = pd.DataFrame({
                'Feature': feature_names,
                'PC1 Loading': pca_model.components_[0],
                'PC2 Loading': pca_model.components_[1]
            })
            
            st.dataframe(loadings_df.head(10), use_container_width=True)
    
    # ========================================================================
    # TAB 5: FEATURE IMPORTANCE
    # ========================================================================
    with tab5:
        st.subheader("🎯 Feature Importance Analysis")
        
        # Get feature importance
        importance_df, rf_model = get_feature_importance(df)
        st.session_state.feature_importance = importance_df
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Feature importance bar chart
            fig_importance = px.bar(
                importance_df.head(10),
                x='Importance',
                y='Feature',
                orientation='h',
                title='Top 10 Feature Importance (Random Forest)',
                color='Importance',
                color_continuous_scale='Viridis',
                labels={'Importance': 'Feature Importance', 'Feature': 'Features'}
            )
            fig_importance.update_layout(height=400)
            st.plotly_chart(fig_importance, use_container_width=True)
        
        with col2:
            st.markdown("### Top 5 Features")
            for idx, row in importance_df.head(5).iterrows():
                st.metric(row['Feature'], f"{row['Importance']:.3f}")
        
        # SHAP-like feature analysis
        with st.expander("🔬 Feature Impact Analysis"):
            st.markdown("#### Feature Directional Impact")
            
            # Calculate feature impact direction
            impact_data = []
            for feature in importance_df['Feature'].head(5):
                # Calculate correlation with target
                if feature in df.columns:
                    corr_val = df[feature].corr(df['Heart Disease'])
                    impact_data.append({
                        'Feature': feature,
                        'Correlation with Disease': corr_val,
                        'Impact Direction': 'Positive' if corr_val > 0 else 'Negative'
                    })
            
            impact_df = pd.DataFrame(impact_data)
            st.dataframe(impact_df, use_container_width=True)
    
    # ========================================================================
    # TAB 6: DATA PREVIEW
    # ========================================================================
    with tab6:
        st.subheader("📋 Data Preview")
        
        # Show data stats
        st.markdown("### Dataset Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Numeric Features Summary**")
            st.dataframe(df.describe(), use_container_width=True)
        
        with col2:
            st.write("**Dataset Info**")
            st.write(f"- **Shape:** {df.shape}")
            st.write(f"- **Missing Values:** {df.isnull().sum().sum()}")
            st.write(f"- **Memory Usage:** {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
        
        # Data sample
        st.markdown("### Data Sample")
        st.dataframe(df.head(20), use_container_width=True)
        
        # Download button
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Dataset as CSV",
            data=csv,
            file_name='heart_disease_data.csv',
            mime='text/csv',
        )

# ============================================================================
# SIDEBAR
# ============================================================================
def show_sidebar():
    """Display sidebar with additional controls and information"""
    st.sidebar.header("🎮 Controls")
    
    # Analysis options
    st.sidebar.subheader("Analysis Settings")
    show_outliers = st.sidebar.checkbox("Show Outlier Detection", value=False)
    show_distribution = st.sidebar.checkbox("Show Distribution Plots", value=True)
    
    st.sidebar.markdown("---")
    
    # Dataset information
    st.sidebar.subheader("📊 Dataset Info")
    df = st.session_state.get('df')
    if df is not None:
        st.sidebar.write(f"**Records:** {len(df)}")
        st.sidebar.write(f"**Features:** {len(df.columns)}")
        
        # Feature types
        numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
        categorical_cols = len(df.select_dtypes(include=['object']).columns)
        st.sidebar.write(f"**Numeric:** {numeric_cols}")
        st.sidebar.write(f"**Categorical:** {categorical_cols}")
    
    st.sidebar.markdown("---")
    
    # About section
    st.sidebar.subheader("ℹ️ About")
    st.sidebar.info("""
    This application provides advanced analytics for heart disease data using:
    - Interactive visualizations with Plotly
    - Machine learning for feature importance
    - PCA for dimensionality reduction
    - Comprehensive statistical analysis
    """)

# ============================================================================
# MAIN APPLICATION
# ============================================================================
if __name__ == '__main__':
    # Page config
    st.set_page_config(
        page_title="Heart Disease Data Explorer",
        page_icon="❤️",
        layout="wide"
    )
    
    # Show sidebar
    show_sidebar()
    
    # Main content
    show_explore_page()
