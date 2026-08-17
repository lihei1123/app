import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer, Dense, MultiHeadAttention, LayerNormalization, Dropout
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
import joblib
import xgboost as xgb
import lightgbm as lgb

# Set page configuration
st.set_page_config(page_title="Online Survival Risk Prediction Platform for Liver Cancer Patients", layout="wide")

# --- 1. Define TransformerEncoder Layer (Must match training) ---
class TransformerEncoder(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"), 
            Dense(embed_dim)
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super(TransformerEncoder, self).get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "rate": self.rate,
        })
        return config


# --- Compatibility wrappers for RNN layers ---
# Some models were saved with a `time_major` argument in layer configs.
# Certain TensorFlow/Keras versions don't accept this keyword when
# reconstructing layers from config. These small wrappers pop that
# argument if present so `load_model` can succeed.
class SimpleRNNCompat(tf.keras.layers.SimpleRNN):
    def __init__(self, *args, **kwargs):
        kwargs.pop('time_major', None)
        super(SimpleRNNCompat, self).__init__(*args, **kwargs)


class LSTMCompat(tf.keras.layers.LSTM):
    def __init__(self, *args, **kwargs):
        kwargs.pop('time_major', None)
        super(LSTMCompat, self).__init__(*args, **kwargs)


class GRUCompat(tf.keras.layers.GRU):
    def __init__(self, *args, **kwargs):
        kwargs.pop('time_major', None)
        super(GRUCompat, self).__init__(*args, **kwargs)

# --- 2. Feature Definition ---
SELECTED_FEATURES = [
    'CA199', 'PLR', 'Tumor differentiation', 'Age', 'GGT',
    'BMI', 'CEA', 'ALT', 'Peroperative bleeding', 'NLR', 'TBIL',
    'Tumor number', 'Lymph node dissection', 'Hepatocirrhosis',
    'ALB', 'Nerve invasion', 'PT', 'Tumor size', 'MVI', 'AJCC stage',
    'AST', 'Lymphatic metastasis', 'Gender', 'Liver caosule invasion',
    'Extent of liver resection', 'AFP', 'Hepatobiliary disease',
    'Cardiopulmonary disease', 'Surgical type', 'Hepatitis'
]

MODEL_FEATURES = SELECTED_FEATURES + ['AC']

TRADITIONAL_FEATURES = [
    'CA199', 'PLR', 'Tumor differentiation', 'Age', 'GGT',
    'BMI', 'CEA', 'ALT', 'Peroperative bleeding', 'NLR', 'TBIL',
    'Tumor number', 'Lymph node dissection', 'Hepatocirrhosis',
    'ALB', 'Nerve invasion'
]
TRADITIONAL_MODEL_FEATURES = TRADITIONAL_FEATURES + ['AC']

# --- 3. Load Data and Scaler (Cached) ---
@st.cache_resource
def load_data_and_scaler():
    train_path = 'train.xlsx'
    if not os.path.exists(train_path):
        st.error(f"Training data file not found: {train_path}, unable to calculate feature means and standardization parameters.")
        return None, None, None, None

    df_train = pd.read_excel(train_path)
    
    # Extract numerical features for Scaler fitting
    # Note: Assumes column names in train.xlsx match SELECTED_FEATURES
    # If some features are missing in train.xlsx, handle them, but based on 1.py logic, they should exist
    
    # Preprocessing logic reuses parts of 1.py
    # 1.py merges then processes, here simplified: read and process numerical columns directly
    # Ensure columns exist. Fit scaler on MODEL_FEATURES (31 features incl. AC) so
    # internal model input matches what the trained models expect.
    available_features = [f for f in MODEL_FEATURES if f in df_train.columns]
    if len(available_features) < len(MODEL_FEATURES):
        missing = set(MODEL_FEATURES) - set(available_features)
        st.warning(f"Missing features in training data: {missing}")

    X_df = df_train[available_features].apply(pd.to_numeric, errors='coerce')
    
    # Calculate means for filling missing values
    means = X_df.mean()
    
    # Fill missing values in training set to fit Scaler
    X_filled = X_df.fillna(means)
    
    scaler = StandardScaler()
    scaler.fit(X_filled)
    
    # Also return training set survival data for calculating Transformer_MLP baseline survival function
    # Assume train.xlsx has 'OS' (survival time) and 'OS' (death status? to be confirmed)
    # In 1.py: 
    # df['OS_months'] = pd.to_numeric(df['OS'], errors='coerce').clip(0, 180)
    # death_within_2y = (df['OS_months'] <= 24).astype(int) -> This is a binary label
    # y_surv = Surv.from_arrays(event=death_within_2y.values, time=df['OS_months'].values)
    # Note: In 1.py y_surv event is death_within_2y, which is "death within 2 years" event.
    # But usually survival analysis event is "whether endpoint event (death) occurred", time is "observation time".
    # 1.py processing is a bit special, it defines event as (OS_months <= 24).
    # If Transformer_MLP is trained with this y_surv, then its "risk" refers to risk of "death within 2 years"?
    # Or it is actually trained with original event (whether death)?
    # Check 1.py train_models -> RSF part uses y_surv_train.
    # Assume Transformer_MLP is also trained with similar survival data.
    # To plot survival curve, we need (time, event) data.
    # Let's extract time and event following 1.py logic.
    
    df_train['OS_months'] = pd.to_numeric(df_train['OS'], errors='coerce').clip(0, 180)
    # The event definition here is critical. Usually event=1 means death.
    # In 1.py: death_within_2y = (df['OS_months'] <= 24).astype(int)
    # This seems to be defined for binary classification task.
    # If Transformer_MLP is a survival model (like DeepSurv), it should use (time, status) where status=1 is death.
    # Let's assume train.xlsx has a column representing status, or we infer from OS_months and some status column.
    # 1.py does not explicitly read 'Status' column, but directly uses (OS_months <= 24) as event.
    # This means the model predicts "whether death within 2 years" survival model? This is a bit strange.
    # But for consistency, we use y_surv definition in 1.py.
    
    event = (df_train['OS_months'] <= 24).astype(int)
    time = df_train['OS_months'].values
    
    # Return: scaler, means(Series), X_train_processed(for calculating baseline hazard), time, event
    return scaler, means, scaler.transform(X_filled), time, event

# --- 4. Load Models (Cached) ---
@st.cache_resource
def load_models():
    models = {}
    model_paths = {
        'MLP': 'best_MLP_final6605.h5',
        'Transformer_MLP': 'best_transformer_mlp_survival_final.h5',
        'Transformer': 'best_Transformer_final7034.h5',
        'RNN': 'best_RNN_final6598.h5',
        'LSTM': 'best_LSTM_final6842.h5',
        'GRU': 'best_GRU_final6762.h5'
    }
    
    # Load Deep Learning Models
    # Prepare compatibility mapping for custom_objects
    compat_map = {
        'TransformerEncoder': TransformerEncoder,
        'SimpleRNN': SimpleRNNCompat,
        'LSTM': LSTMCompat,
        'GRU': GRUCompat
    }

    for name, path in model_paths.items():
        if os.path.exists(path):
            try:
                # Transformer related models need custom_objects
                # Use compat_map for all models so RNN/LSTM/GRU layers with
                # legacy kwargs like `time_major` are accepted during load.
                model = load_model(path, custom_objects=compat_map, compile=False)
                models[name] = model
            except Exception as e:
                st.error(f"Failed to load model {name}: {e}")
        else:
            st.warning(f"Model file does not exist: {path}")

    # Load Traditional Machine Learning Models
    # Assume these models are saved as .joblib files (based on model types in 1-2.py)
    traditional_paths = {
        'SVM': 'svm_model.joblib',
        'Logistic Regression': 'logistic_regression.joblib',
        'XGBoost': 'xgboost.joblib',
        'Random Forest': 'random_forest.joblib',
        'LightGBM': 'lightgbm.joblib'
    }

    for name, path in traditional_paths.items():
        if os.path.exists(path):
            try:
                model = joblib.load(path)
                models[name] = model
            except Exception as e:
                st.error(f"Failed to load traditional model {name}: {e}")
        # else:
            # st.warning(f"Traditional model file does not exist: {path} (might not be trained/saved yet)")
            
    return models

# --- 5. Calculate Baseline Survival Function (Breslow Estimator) ---
def calculate_baseline_survival(model, X_train, time_train, event_train):
    """
    Calculate baseline survival function S0(t) using Breslow estimator.
    Assumes model output is log-hazard (risk score).
    """
    # Predict risk scores for training set
    # Note: Transformer_MLP output shape might need adjustment
    try:
        risk_scores = model.predict(X_train, verbose=0).flatten()
    except:
        # Try adjusting input shape
        if X_train.ndim == 2:
            X_train_3d = np.expand_dims(X_train, axis=1)
            risk_scores = model.predict(X_train_3d, verbose=0).flatten()
        else:
            risk_scores = np.zeros(len(X_train)) # Fallback

    # Create DataFrame for easier processing
    df = pd.DataFrame({
        'time': time_train,
        'event': event_train,
        'risk': risk_scores
    })
    
    # Sort by time
    df = df.sort_values(by='time')
    
    unique_times = df['time'].unique()
    unique_times.sort()
    
    baseline_hazard = []
    cumulative_hazard = 0
    
    for t in unique_times:
        # Number of events at time t
        d_t = df[(df['time'] == t) & (df['event'] == 1)].shape[0]
        
        # Risk set at time t (time >= t)
        risk_set = df[df['time'] >= t]
        
        # Sum of exp(risk) for all individuals in risk set
        sum_exp_risk = np.sum(np.exp(risk_set['risk']))
        
        if sum_exp_risk > 0:
            hazard_contribution = d_t / sum_exp_risk
        else:
            hazard_contribution = 0
            
        cumulative_hazard += hazard_contribution
        baseline_hazard.append({'time': t, 'H0': cumulative_hazard})
        
    baseline_df = pd.DataFrame(baseline_hazard)
    # S0(t) = exp(-H0(t))
    baseline_df['S0'] = np.exp(-baseline_df['H0'])
    
    return baseline_df

# --- 6. Main Program ---
def main():
    # Custom CSS styles
    st.markdown("""
        <style>
        .main {
            background-color: #f8f9fa;
        }
        .stButton>button {
            width: 100%;
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
            border-radius: 5px;
            height: 3em;
        }
        .metric-card {
            background-color: white;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }
        h1 {
            color: #2c3e50;
            text-align: center;
            padding-bottom: 20px;
        }
        h3 {
            color: #34495e;
            border-left: 5px solid #4CAF50;
            padding-left: 10px;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("🏥 Online Survival Risk Prediction Platform for Liver Cancer Patients")
    st.markdown("""
    <div style='text-align: center; color: #666; margin-bottom: 30px;'>
    Intelligent Computer-Aided Diagnosis System Based on Multimodal Deep Learning and Traditional Machine Learning Algorithms
    </div>
    """, unsafe_allow_html=True)
    
    # 加载资源
    scaler, means, X_train_processed, time_train, event_train = load_data_and_scaler()
    models = load_models()
    
    if scaler is None or not models:
        st.error("System initialization failed. Please check data and model files.")
        return

    st.sidebar.header("📋 Patient Feature Input")
    st.sidebar.info("Please enter the patient's clinical feature data below. Unfilled items will be automatically filled with the training set mean.")
    
    user_inputs = {}
    
    # 使用 Expander 分组显示特征，避免侧边栏过长
    with st.sidebar.form("feature_form"):
        # 将特征分为几组显示
        groups = np.array_split(SELECTED_FEATURES, 5)
        
        # 特征默认值设置为0的列表
        ZERO_DEFAULT_FEATURES = [
            'Gender', 'Hepatitis', 'Hepatocirrhosis', 'Hepatobiliary disease',
            'Cardiopulmonary disease', 'Surgical type', 'Lymph node dissection',
            'Extent of liver resection', 'AFP', 'CEA', 'CA199',
            'Tumor differentiation', 'MVI', 'Nerve invasion',
            'Liver caosule invasion', 'Tumor number', 'Tumor size'
        ]

        for i, group in enumerate(groups):
            st.markdown(f"**Feature Group {i+1}**")
            cols = st.columns(2)
            for idx, feature in enumerate(group):
                # 获取默认值
                if feature in ZERO_DEFAULT_FEATURES:
                    default_val = 0.0
                else:
                    default_val = float(means[feature]) if feature in means else 0.0
                
                # 创建输入框
                with cols[idx % 2]:
                    user_inputs[feature] = st.number_input(
                        f"{feature}", 
                        value=default_val,
                        format="%.4f"
                    )
        
        submitted = st.form_submit_button("🚀 Start Prediction")

    if submitted:
        # 1. 数据预处理
        # Build a 30-col user input frame (no AC) then immediately pad AC with
        # the training-set mean so the scaler (fit on 31 cols) sees the same
        # columns it was trained on.
        user_only_df = pd.DataFrame([user_inputs])[SELECTED_FEATURES]

        ac_default = float(means['AC']) if 'AC' in means.index else 0.0
        full_input_df = user_only_df.copy()
        full_input_df['AC'] = ac_default
        full_input_df = full_input_df[MODEL_FEATURES]  # 31 cols, exact scaler order

        X_scaled_full = scaler.transform(full_input_df)
        X_scaled_df = pd.DataFrame(X_scaled_full, columns=MODEL_FEATURES)

        # UI-facing 30-dim scaled slice (no AC) for downstream statistics / plots
        X_scaled = X_scaled_df[SELECTED_FEATURES].values

        # Internal model-facing features
        X_full = X_scaled_df[MODEL_FEATURES].values        # (1, 31) for DL models
        X_trad = X_scaled_df[TRADITIONAL_MODEL_FEATURES].values  # (1, 17) for trad ML

        # 2. 模型预测
        results = []

        # 预先计算 Transformer_MLP 的基线生存
        baseline_df = None
        if 'Transformer_MLP' in models:
            with st.spinner("Calculating survival curve baseline..."):
                baseline_df = calculate_baseline_survival(models['Transformer_MLP'], X_train_processed, time_train, event_train)

        dl_models = ['Transformer', 'Transformer_MLP', 'MLP', 'RNN', 'LSTM', 'GRU']
        trad_models = ['Logistic Regression', 'XGBoost', 'Random Forest', 'SVM', 'LightGBM']
        sorted_model_names = [m for m in dl_models if m in models] + [m for m in trad_models if m in models]

        # 收集预测结果
        predictions_2y = {} # 存储2年生存概率用于绘图

        for name in sorted_model_names:
            model = models[name]

            # --- 传统机器学习模型处理 ---
            if name in trad_models:
                X_in = X_trad
                try:
                    if hasattr(model, "predict_proba"):
                        pred = model.predict_proba(X_in)[:, 1][0]
                    else:
                        pred = model.predict(X_in)[0]
                    
                    surv_prob_2y = pred
                    risk_prob = 1 - pred
                    predictions_2y[name] = surv_prob_2y
                    
                    results.append({
                        "Model": name,
                        "Type": "Traditional ML",
                        "2-Year Survival Prob": f"{surv_prob_2y:.2%}",
                        "Risk Index": f"{risk_prob:.4f}"
                    })
                except Exception as e:
                    st.warning(f"Model {name} prediction failed: {e}")
                continue

            # --- 深度学习模型处理 ---
            X_in = X_full
            if hasattr(model, 'input_shape') and len(model.input_shape) == 3:
                 X_in = np.expand_dims(X_full, axis=1)
            
            pred = model.predict(X_in, verbose=0).flatten()[0]
            
            if name == 'Transformer_MLP':
                risk_score = pred
                if baseline_df is not None:
                    s0_24 = baseline_df.iloc[(baseline_df['time'] - 24).abs().argsort()[:1]]['S0'].values[0]
                    surv_prob_2y = np.power(s0_24, np.exp(risk_score))
                else:
                    surv_prob_2y = np.nan
                
                predictions_2y[name] = surv_prob_2y
                results.append({
                    "Model": name,
                    "Type": "Deep Learning (Survival)",
                    "2-Year Survival Prob": f"{surv_prob_2y:.2%}" if not np.isnan(surv_prob_2y) else "N/A",
                    "Risk Index": f"{risk_score:.4f} "
                })
                
            else:
                surv_prob_2y = pred
                risk_prob = 1 - pred
                predictions_2y[name] = surv_prob_2y
                
                results.append({
                    "Model": name,
                    "Type": "Deep Learning (Binary)",
                    "2-Year Survival Prob": f"{surv_prob_2y:.2%}",
                    "Risk Index": f"{risk_prob:.4f}"
                })

        # --- 仪表盘展示 ---
        
        # 计算平均生存概率
        valid_probs = [p for p in predictions_2y.values() if not np.isnan(p)]
        avg_prob = np.mean(valid_probs) if valid_probs else 0
        
        st.markdown("### 📊 Core Risk Assessment")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Models Run", f"{len(results)}")
        with col2:
            st.metric("Avg 2-Year Survival Rate", f"{avg_prob:.1%}", delta_color="normal")
        with col3:
            risk_level = "Low Risk" if avg_prob > 0.7 else "Medium Risk" if avg_prob > 0.4 else "High Risk"
            st.metric("Comprehensive Risk Level", risk_level, delta="-High Risk" if risk_level=="High Risk" else "off")

        # 使用自定义HTML分割线来调整间距 (margin-top: -20px 减小与上方指标的距离)
        st.markdown("""<hr style="margin-top: -5px; margin-bottom: 20px;" />""", unsafe_allow_html=True)

        # 使用 Tabs 分页展示详情
        tab1, tab2, tab3 = st.tabs(["📈 Survival Curve Analysis", "🤖 Model Prediction Comparison", "👤 Patient Feature Profile"])

        with tab1:
            st.subheader("Individualized Survival Curve Prediction")
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # 绘制 Transformer_MLP 曲线
            if 'Transformer_MLP' in models and baseline_df is not None:
                # 获取风险分
                risk_score = [r['Risk Index'] for r in results if r['Model']=='Transformer_MLP'][0]
                # 解析回数值 (去掉字符串中的说明)
                risk_score = float(risk_score.split()[0])
                patient_surv = np.power(baseline_df['S0'], np.exp(risk_score))
                ax.step(baseline_df['time'], patient_surv, where="post", label="Transformer_MLP Survival Curve", linewidth=3, color='#e74c3c')
            
            # 绘制其他模型的点估计
            for name, prob in predictions_2y.items():
                if name != 'Transformer_MLP':
                    marker = 's' if name in trad_models else 'o'
                    color = '#3498db' if name in trad_models else '#2ecc71'
                    ax.plot(24, prob, marker=marker, markersize=8, linestyle='', label=f"{name}", color=color, alpha=0.7)

            ax.set_xlabel("Time (Months)", fontsize=12)
            ax.set_ylabel("Survival Probability", fontsize=12)
            ax.set_title("Patient Specific Survival Analysis", fontsize=14)
            ax.set_ylim(0, 1.05)
            ax.axvline(x=24, color='gray', linestyle='--', alpha=0.5, label="24 Months Threshold")
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            st.markdown("#### Detailed Prediction Data")
            st.dataframe(pd.DataFrame(results).style.highlight_max(axis=0, subset=['2-Year Survival Prob'], color='#d4edda'))

        with tab2:
            st.subheader("Comparison of 2-Year Survival Rate Predictions by Model")
            
            # 准备数据
            model_names = list(predictions_2y.keys())
            probs = list(predictions_2y.values())
            colors = ['#e74c3c' if 'Transformer' in m else '#3498db' if m in trad_models else '#2ecc71' for m in model_names]
            
            fig2, ax2 = plt.subplots(figsize=(10, len(model_names)*0.5 + 2))
            y_pos = np.arange(len(model_names))
            
            ax2.barh(y_pos, probs, align='center', color=colors, alpha=0.8)
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(model_names)
            ax2.invert_yaxis()  # labels read top-to-bottom
            ax2.set_xlabel('2-Year Survival Probability')
            ax2.set_title('Model Comparison')
            ax2.set_xlim(0, 1.0)
            ax2.grid(axis='x', alpha=0.3)
            
            # 添加数值标签
            for i, v in enumerate(probs):
                ax2.text(v + 0.01, i, f"{v:.1%}", va='center', fontweight='bold')
                
            st.pyplot(fig2)
            st.caption("Red: Transformer Series | Green: Other Deep Learning | Blue: Traditional Machine Learning")

        with tab3:
            st.subheader("Patient Feature Deviation Analysis (Z-Score)")
            st.markdown("Shows the deviation of patient feature values from the training set mean (in standard deviations).")
            
            # 计算 Z-scores (即 X_scaled)
            # 选取传统模型使用的重要特征进行展示，避免过多
            feat_subset = TRADITIONAL_FEATURES
            # 找到这些特征在 SELECTED_FEATURES 中的索引
            indices = [SELECTED_FEATURES.index(f) for f in feat_subset]
            z_scores = X_scaled[0][indices]
            
            # 绘图
            fig3, ax3 = plt.subplots(figsize=(10, 8))
            y_pos = np.arange(len(feat_subset))
            
            # 根据正负值着色
            bar_colors = ['#ff9999' if x > 0 else '#66b3ff' for x in z_scores]
            
            ax3.barh(y_pos, z_scores, align='center', color=bar_colors, alpha=0.8)
            ax3.set_yticks(y_pos)
            ax3.set_yticklabels(feat_subset)
            ax3.invert_yaxis()
            ax3.set_xlabel('Z-Score (Standard Deviations from Mean)')
            ax3.set_title('Feature Deviation Profile')
            ax3.axvline(x=0, color='black', linestyle='-', linewidth=1)
            ax3.grid(axis='x', alpha=0.3)
            
            st.pyplot(fig3)
            st.caption("Red indicates above average, Blue indicates below average.")

if __name__ == "__main__":
    main()
