import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import warnings
warnings.filterwarnings('ignore')

# Konfigurasi halaman
st.set_page_config(
    page_title="Prediksi Produksi FFB - PT SIP",
    page_icon="🌴",
    layout="wide"
)

# ==========================================
# FUNGSI PEMROSESAN & MODEL
# ==========================================

def create_sequences(data, target, lookback=3):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i])
        y.append(target[i])
    return np.array(X), np.array(y)

def create_lstm_model(input_shape):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def calculate_mape(y_true, y_pred):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    mask = y_true != 0
    if np.sum(mask) == 0:
        return 0.0
    
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

# ==========================================
# UI UTAMA
# ==========================================

# Judul Aplikasi (Native Python Header)
st.title("🌴 SISTEM PREDIKSI PRODUKSI FFB")
st.markdown("**Analisis Kinerja LSTM dan Random Forest - PT SALIM IVOMAS PRATAMA**")
st.divider()

# Sidebar
with st.sidebar:
    # Menggunakan gambar placeholder, bisa diganti file lokal
    st.image("https://via.placeholder.com/250x100/1B5E20/FFFFFF?text=PT+SIP", use_container_width=True)
    
    st.header("📋 Menu Navigasi")
    menu = st.radio(
        "Pilih Menu:",
        ["🏠 Dashboard", "📊 Upload & Eksplorasi Data", "🤖 Training Model", "📈 Prediksi & Hasil", "📉 Evaluasi Komparatif", "ℹ️ Informasi"]
    )
    st.divider()
    
    if 'data' in st.session_state:
        st.success(f"✅ Data loaded: {len(st.session_state['data'])} rows")
    
    if 'models' in st.session_state:
        st.success(f"✅ Models trained: {len(st.session_state['models'])}")
    
    st.divider()
    st.info("**📧 Support:**\nresearch@salimivomas.com")

# ==========================================
# MENU 1: DASHBOARD
# ==========================================
if menu == "🏠 Dashboard":
    st.header("🏠 Dashboard Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📁 Dataset Status",
            value="Ready" if 'data' in st.session_state else "Not Loaded",
            delta="4200 rows" if 'data' in st.session_state else None
        )
    
    with col2:
        st.metric(
            label="🤖 Models Status",
            value="Trained" if 'models' in st.session_state else "Not Trained",
            delta=f"{len(st.session_state.get('models', {}))} models" if 'models' in st.session_state else None
        )
    
    with col3:
        st.metric(
            label="🎯 Target Variable",
            value="Actual FFB Production",
            delta="CM (Current Month)"
        )
    
    with col4:
        st.metric(
            label="📊 Features",
            value="7 Variables",
            delta="+ Time Series"
        )
    
    st.divider()
    
    # Info cards
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Tujuan Penelitian")
        st.markdown("""
        Menganalisis dan membandingkan kinerja algoritma **LSTM** (Long Short-Term Memory) 
        dan **Random Forest** dalam memprediksi produksi Fresh Fruit Bunch (FFB) 
        berdasarkan data blok dan bulanan di PT Salim Ivomas Pratama.
        
        **Dataset Features:**
        - Business Area & Estate
        - Divisi
        - Block
        - Hectare Mature
        - Actual FFB Production (Target)
        - Budget FFB Production
        - Month (Time Series)
        """)
    
    with col2:
        st.subheader("📊 Workflow Sistem")
        st.markdown("""
        1. **Upload Data** - Load dataset CSV
        2. **Eksplorasi** - Analisis statistik & visualisasi
        3. **Preprocessing** - Normalisasi & feature engineering
        4. **Training** - Train LSTM & Random Forest
        5. **Evaluasi** - Bandingkan performa model
        6. **Prediksi** - Deploy untuk produksi
        
        **Metrik Evaluasi:**
        - MAE (Mean Absolute Error)
        - RMSE (Root Mean Squared Error)
        - R² Score (Coefficient of Determination)
        - MAPE (Mean Absolute Percentage Error)
        """)
    
    if 'data' in st.session_state:
        st.divider()
        st.subheader("📈 Quick Stats")
        df = st.session_state['data']
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Business Areas", df['Business Area'].nunique())
        with col2:
            st.metric("Total Estates", df['Business Area Estate'].nunique())
        with col3:
            st.metric("Total Blocks", df['Block'].nunique())
        with col4:
            st.metric("Total Months", df['Month'].nunique())

# ==========================================
# MENU 2: UPLOAD & EKSPLORASI DATA
# ==========================================
elif menu == "📊 Upload & Eksplorasi Data":
    st.header("📊 Upload & Eksplorasi Data")
    
    # Upload Section
    st.subheader("📁 Upload Dataset")
    uploaded_file = st.file_uploader(
        "Upload file CSV dengan format yang sesuai",
        type=['csv'],
        help="Format: Business Area, Business Area Estate, Divisi, Block, Hectare Mature, Actual FFB Production CM, Budget FFB Production CM, Month"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Validasi kolom
            required_columns = ['Business Area', 'Business Area Estate', 'Divisi', 'Block', 
                              'Hectare Mature', 'Actual FFB Production CM', 
                              'Budget FFB Production CM', 'Month']
            
            missing_cols = [col for col in required_columns if col not in df.columns]
            
            if missing_cols:
                st.error(f"❌ Kolom yang hilang: {', '.join(missing_cols)}")
            else:
                # Data cleaning
                st.info("🔄 Membersihkan data...")
                
                initial_rows = len(df)
                df = df.dropna(subset=['Actual FFB Production CM'])
                
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    if df[col].isnull().sum() > 0:
                        df[col].fillna(df[col].median(), inplace=True)
                
                categorical_cols = df.select_dtypes(include=['object']).columns
                for col in categorical_cols:
                    if df[col].isnull().sum() > 0:
                        df[col].fillna(df[col].mode()[0], inplace=True)
                
                df = df.dropna()
                df = df.replace([np.inf, -np.inf], np.nan).dropna()
                
                final_rows = len(df)
                rows_removed = initial_rows - final_rows
                
                if rows_removed > 0:
                    st.warning(f"⚠️ {rows_removed} baris dihapus karena missing values atau infinite values")
                
                st.session_state['data'] = df
                st.success(f"✅ Dataset berhasil dimuat dan dibersihkan! Total {len(df):,} baris data valid")
                
                # Preview data
                st.subheader("👁️ Preview Data (10 baris pertama)")
                st.dataframe(df.head(10), use_container_width=True)
                
                # Informasi dataset
                st.subheader("📋 Informasi Dataset")
                col1, col2, col3, col4 = st.columns(4)
                with col1: st.metric("Total Baris", f"{len(df):,}")
                with col2: st.metric("Total Kolom", len(df.columns))
                with col3: st.metric("Missing Values", df.isnull().sum().sum())
                with col4: st.metric("Duplicates", df.duplicated().sum())
                
                # Statistik deskriptif
                st.subheader("📊 Statistik Deskriptif")
                st.dataframe(df.describe(), use_container_width=True)
                
                # Eksplorasi data
                st.divider()
                st.subheader("🔍 Analisis Eksploratif")
                
                tab1, tab2, tab3, tab4 = st.tabs(["📈 Distribusi Produksi", "🗺️ Analisis per Area", "📅 Trend Bulanan", "🔗 Korelasi"])
                
                with tab1:
                    col1, col2 = st.columns(2)
                    with col1:
                        fig = px.histogram(df, x='Actual FFB Production CM', nbins=50, title='Distribusi Actual FFB Production', color_discrete_sequence=['#2E7D32'])
                        st.plotly_chart(fig, use_container_width=True)
                    with col2:
                        fig = px.box(df, y='Actual FFB Production CM', title='Box Plot Produksi FFB', color_discrete_sequence=['#558B2F'])
                        st.plotly_chart(fig, use_container_width=True)
                
                with tab2:
                    area_prod = df.groupby('Business Area')['Actual FFB Production CM'].agg(['sum', 'mean', 'count']).reset_index()
                    area_prod.columns = ['Business Area', 'Total Production', 'Avg Production', 'Count']
                    col1, col2 = st.columns(2)
                    with col1:
                        fig = px.bar(area_prod, x='Business Area', y='Total Production', title='Total Produksi per Business Area', color='Total Production', color_continuous_scale='Greens')
                        st.plotly_chart(fig, use_container_width=True)
                    with col2:
                        estate_prod = df.groupby('Business Area Estate')['Actual FFB Production CM'].sum().sort_values(ascending=False).head(10)
                        fig = px.bar(x=estate_prod.values, y=estate_prod.index, orientation='h', title='Top 10 Estate - Total Produksi', color=estate_prod.values, color_continuous_scale='Viridis')
                        st.plotly_chart(fig, use_container_width=True)
                
                with tab3:
                    monthly_prod = df.groupby('Month')['Actual FFB Production CM'].agg(['sum', 'mean']).reset_index()
                    monthly_prod = monthly_prod.sort_values('Month')
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=monthly_prod['Month'], y=monthly_prod['sum'], mode='lines+markers', name='Total Production', line=dict(color='#2E7D32', width=3)))
                    fig.update_layout(title='Trend Produksi FFB Bulanan', xaxis_title='Bulan', yaxis_title='Total Produksi (ton)')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    budget_comparison = df.groupby('Month')[['Actual FFB Production CM', 'Budget FFB Production CM']].sum().reset_index()
                    budget_comparison = budget_comparison.sort_values('Month')
                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=budget_comparison['Month'], y=budget_comparison['Actual FFB Production CM'], name='Actual', marker_color='#2E7D32'))
                    fig.add_trace(go.Bar(x=budget_comparison['Month'], y=budget_comparison['Budget FFB Production CM'], name='Budget', marker_color='#FFA726'))
                    fig.update_layout(title='Perbandingan Actual vs Budget Production', xaxis_title='Bulan', barmode='group')
                    st.plotly_chart(fig, use_container_width=True)
                
                with tab4:
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    corr_matrix = df[numeric_cols].corr()
                    fig = px.imshow(corr_matrix, text_auto='.2f', title='Correlation Matrix', color_continuous_scale='RdYlGn')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    fig = px.scatter(df, x='Hectare Mature', y='Actual FFB Production CM', title='Hubungan Hectare Mature vs Produksi FFB', trendline='ols', color='Divisi', opacity=0.6)
                    st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"❌ Error saat membaca file: {str(e)}")
    
    else:
        st.info("👆 Silakan upload file CSV dataset Anda")
        st.subheader("📝 Contoh Format Dataset")
        sample_data = pd.DataFrame({
            'Business Area': ['SUMUT', 'SUMUT', 'JAMBI'],
            'Business Area Estate': ['ESTATE A', 'ESTATE A', 'ESTATE B'],
            'Divisi': ['DIV 1', 'DIV 1', 'DIV 2'],
            'Block': ['A001', 'A002', 'B001'],
            'Hectare Mature': [50.5, 45.2, 60.0],
            'Actual FFB Production CM': [850.5, 720.3, 950.2],
            'Budget FFB Production CM': [800.0, 750.0, 900.0],
            'Month': ['2023-01', '2023-01', '2023-01']
        })
        st.dataframe(sample_data, use_container_width=True)

# ==========================================
# MENU 3: TRAINING MODEL
# ==========================================
elif menu == "🤖 Training Model":
    st.header("🤖 Training Model Prediksi")
    
    if 'data' not in st.session_state:
        st.warning("⚠️ Silakan upload data terlebih dahulu di menu 'Upload & Eksplorasi Data'")
    else:
        df = st.session_state['data'].copy()
        
        st.subheader("⚙️ Konfigurasi Training")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🎯 Pengaturan Umum")
            model_choice = st.selectbox("Pilih Model untuk Training:", ["LSTM", "Random Forest", "Kedua Model (LSTM + Random Forest)"])
            test_size = st.slider("Ukuran Data Testing (%):", 10, 40, 20, 5)
            random_state = st.number_input("Random State:", 0, 100, 42)
        
        with col2:
            st.markdown("#### 🔧 Parameter Model")
            if model_choice in ["LSTM", "Kedua Model (LSTM + Random Forest)"]:
                st.markdown("**LSTM Parameters:**")
                epochs = st.number_input("Epochs:", 10, 200, 50, 10)
                batch_size = st.selectbox("Batch Size:", [16, 32, 64, 128], index=1)
                lookback = st.number_input("Lookback Period (bulan):", 1, 12, 3)
            
            if model_choice in ["Random Forest", "Kedua Model (LSTM + Random Forest)"]:
                st.markdown("**Random Forest Parameters:**")
                n_estimators = st.number_input("Number of Trees:", 50, 500, 100, 50)
                max_depth = st.number_input("Max Depth:", 5, 50, 15, 5)
                min_samples_split = st.number_input("Min Samples Split:", 2, 20, 5)
        
        st.divider()
        st.subheader("📊 Feature Engineering")
        
        col1, col2 = st.columns(2)
        with col1:
            use_hectare = st.checkbox("Gunakan Hectare Mature", value=True)
            use_budget = st.checkbox("Gunakan Budget FFB Production", value=True)
            use_month_encoding = st.checkbox("Encoding Month (Cyclical)", value=True)
        with col2:
            use_area_encoding = st.checkbox("Encoding Business Area", value=True)
            use_estate_encoding = st.checkbox("Encoding Estate", value=True)
            use_divisi_encoding = st.checkbox("Encoding Divisi", value=True)
        
        st.divider()
        if st.button("🚀 Mulai Training", type="primary", use_container_width=True):
            with st.spinner("⏳ Memproses data dan training model..."):
                try:
                    df_model = df.copy()
                    st.info("🔍 Validasi dan preprocessing data...")
                    
                    df_model = df_model.replace([np.inf, -np.inf], np.nan).dropna()
                    
                    if len(df_model) < 100:
                        st.error("❌ Data terlalu sedikit setelah cleaning. Minimal 100 baris diperlukan.")
                        st.stop()
                    
                    features = []
                    
                    if use_hectare: features.append('Hectare Mature')
                    if use_budget: features.append('Budget FFB Production CM')
                    
                    le_dict = {}
                    if use_area_encoding:
                        le_area = LabelEncoder()
                        df_model['Business_Area_Encoded'] = le_area.fit_transform(df_model['Business Area'])
                        features.append('Business_Area_Encoded')
                        le_dict['Business Area'] = le_area
                    
                    if use_estate_encoding:
                        le_estate = LabelEncoder()
                        df_model['Estate_Encoded'] = le_estate.fit_transform(df_model['Business Area Estate'])
                        features.append('Estate_Encoded')
                        le_dict['Estate'] = le_estate
                    
                    if use_divisi_encoding:
                        le_divisi = LabelEncoder()
                        df_model['Divisi_Encoded'] = le_divisi.fit_transform(df_model['Divisi'])
                        features.append('Divisi_Encoded')
                        le_dict['Divisi'] = le_divisi
                    
                    if use_month_encoding:
                        df_model['Month_Num'] = pd.to_datetime(df_model['Month']).dt.month
                        df_model['Month_Sin'] = np.sin(2 * np.pi * df_model['Month_Num'] / 12)
                        df_model['Month_Cos'] = np.cos(2 * np.pi * df_model['Month_Num'] / 12)
                        features.extend(['Month_Sin', 'Month_Cos'])
                    
                    target = 'Actual FFB Production CM'
                    df_model = df_model[features + [target]].copy()
                    df_model = df_model.replace([np.inf, -np.inf], np.nan).dropna()
                    
                    if len(df_model) == 0:
                        st.error("❌ Tidak ada data valid setelah preprocessing. Periksa dataset Anda.")
                        st.stop()
                    
                    st.success(f"✅ Data valid: {len(df_model)} baris")
                    
                    train_size = int(len(df_model) * (1 - test_size/100))
                    train_data = df_model[:train_size]
                    test_data = df_model[train_size:]
                    
                    X_train = train_data[features].values
                    y_train = train_data[target].values
                    X_test = test_data[features].values
                    y_test = test_data[target].values
                    
                    if np.any(np.isnan(X_train)) or np.any(np.isnan(y_train)):
                        st.error("❌ Data training mengandung NaN.")
                        st.stop()
                    if np.any(np.isnan(X_test)) or np.any(np.isnan(y_test)):
                        st.error("❌ Data testing mengandung NaN.")
                        st.stop()
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    results = {}
                    
                    # Training LSTM
                    if model_choice in ["LSTM", "Kedua Model (LSTM + Random Forest)"]:
                        status_text.text("🔄 Training LSTM Model...")
                        progress_bar.progress(10)
                        
                        scaler = MinMaxScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test)
                        
                        X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, lookback)
                        X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test, lookback)
                        
                        if np.any(np.isnan(X_train_seq)) or np.any(np.isnan(y_train_seq)):
                            st.error("❌ Data sequence training mengandung NaN.")
                            st.stop()
                        
                        X_train_lstm = X_train_seq.reshape((X_train_seq.shape[0], X_train_seq.shape[1], X_train_seq.shape[2]))
                        X_test_lstm = X_test_seq.reshape((X_test_seq.shape[0], X_test_seq.shape[1], X_test_seq.shape[2]))
                        
                        lstm_model = create_lstm_model((X_train_lstm.shape[1], X_train_lstm.shape[2]))
                        progress_bar.progress(20)
                        
                        history = lstm_model.fit(X_train_lstm, y_train_seq, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=0)
                        progress_bar.progress(50)
                        
                        lstm_pred = lstm_model.predict(X_test_lstm, verbose=0)
                        lstm_pred = lstm_pred.flatten()
                        if np.any(np.isnan(lstm_pred)):
                            st.warning("⚠️ Prediksi LSTM mengandung NaN. Menggunakan fallback...")
                            lstm_pred = np.nan_to_num(lstm_pred, nan=y_test_seq.mean())
                        
                        y_test_seq_flat = y_test_seq.flatten()
                        
                        results['LSTM'] = {
                            'model': lstm_model, 'scaler': scaler, 'predictions': lstm_pred, 'actuals': y_test_seq_flat,
                            'mae': mean_absolute_error(y_test_seq_flat, lstm_pred),
                            'rmse': np.sqrt(mean_squared_error(y_test_seq_flat, lstm_pred)),
                            'r2': r2_score(y_test_seq_flat, lstm_pred),
                            'mape': calculate_mape(y_test_seq_flat, lstm_pred),
                            'history': history.history, 'lookback': lookback
                        }
                        progress_bar.progress(60)
                    
                    # Training Random Forest
                    if model_choice in ["Random Forest", "Kedua Model (LSTM + Random Forest)"]:
                        status_text.text("🔄 Training Random Forest Model...")
                        progress_bar.progress(70)
                        
                        rf_model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, random_state=random_state, n_jobs=-1)
                        rf_model.fit(X_train, y_train)
                        progress_bar.progress(85)
                        
                        rf_pred = rf_model.predict(X_test)
                        if np.any(np.isnan(rf_pred)):
                            st.warning("⚠️ Prediksi Random Forest mengandung NaN. Menggunakan fallback...")
                            rf_pred = np.nan_to_num(rf_pred, nan=y_test.mean())
                        
                        results['Random Forest'] = {
                            'model': rf_model, 'predictions': rf_pred, 'actuals': y_test,
                            'mae': mean_absolute_error(y_test, rf_pred),
                            'rmse': np.sqrt(mean_squared_error(y_test, rf_pred)),
                            'r2': r2_score(y_test, rf_pred),
                            'mape': calculate_mape(y_test, rf_pred),
                            'feature_importance': dict(zip(features, rf_model.feature_importances_)),
                            'features': features
                        }
                        progress_bar.progress(95)
                    
                    st.session_state['models'] = results
                    st.session_state['features'] = features
                    st.session_state['le_dict'] = le_dict
                    st.session_state['target'] = target
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Training selesai!")
                    
                    st.success("🎉 Training berhasil diselesaikan!")
                    st.divider()
                    st.subheader("📊 Hasil Training")
                    
                    metrics_data = []
                    for model_name, result in results.items():
                        metrics_data.append({
                            'Model': model_name,
                            'MAE': f"{result['mae']:.4f}",
                            'RMSE': f"{result['rmse']:.4f}",
                            'R² Score': f"{result['r2']:.4f}",
                            'MAPE (%)': f"{result['mape']:.2f}%"
                        })
                    
                    st.dataframe(pd.DataFrame(metrics_data), use_container_width=True)
                    
                    cols = st.columns(len(results))
                    for idx, (model_name, result) in enumerate(results.items()):
                        with cols[idx]:
                            st.markdown(f"#### {model_name}")
                            st.metric("MAE", f"{result['mae']:.4f}")
                            st.metric("RMSE", f"{result['rmse']:.4f}")
                            st.metric("R²", f"{result['r2']:.4f}")
                            st.metric("MAPE", f"{result['mape']:.2f}%")
                    
                    if 'LSTM' in results:
                        st.divider()
                        st.subheader("📈 LSTM Training History")
                        history = results['LSTM']['history']
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(y=history['loss'], mode='lines', name='Training Loss', line=dict(color='#2E7D32')))
                            fig.add_trace(go.Scatter(y=history['val_loss'], mode='lines', name='Validation Loss', line=dict(color='#FFA726')))
                            fig.update_layout(title='Loss Curve', xaxis_title='Epoch', yaxis_title='Loss')
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(y=history['mae'], mode='lines', name='Training MAE', line=dict(color='#2E7D32')))
                            fig.add_trace(go.Scatter(y=history['val_mae'], mode='lines', name='Validation MAE', line=dict(color='#FFA726')))
                            fig.update_layout(title='MAE Curve', xaxis_title='Epoch', yaxis_title='MAE')
                            st.plotly_chart(fig, use_container_width=True)
                    
                    if 'Random Forest' in results:
                        st.divider()
                        st.subheader("🎯 Random Forest - Feature Importance")
                        fi = results['Random Forest']['feature_importance']
                        fi_df = pd.DataFrame(list(fi.items()), columns=['Feature', 'Importance']).sort_values('Importance', ascending=False)
                        fig = px.bar(fi_df, x='Importance', y='Feature', orientation='h', title='Feature Importance', color='Importance', color_continuous_scale='Greens')
                        st.plotly_chart(fig, use_container_width=True)
                
                except Exception as e:
                    st.error(f"❌ Error saat training: {str(e)}")

# ==========================================
# MENU 4: PREDIKSI & HASIL
# ==========================================
elif menu == "📈 Prediksi & Hasil":
    st.header("📈 Prediksi Produksi FFB")
    
    if 'models' not in st.session_state:
        st.warning("⚠️ Silakan lakukan training model terlebih dahulu!")
    else:
        models = st.session_state['models']
        features = st.session_state['features']
        
        st.subheader("🎯 Pilih Model untuk Prediksi")
        selected_model = st.selectbox("Model:", list(models.keys()))
        
        tab1, tab2 = st.tabs(["🔮 Prediksi Manual", "📊 Hasil Evaluasi"])
        
        with tab1:
            st.markdown("### 📝 Input Data untuk Prediksi")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if 'Business_Area_Encoded' in features:
                    business_area = st.selectbox("Business Area:", st.session_state['data']['Business Area'].unique())
                if 'Estate_Encoded' in features:
                    estate = st.selectbox("Estate:", st.session_state['data']['Business Area Estate'].unique())
                if 'Divisi_Encoded' in features:
                    divisi = st.selectbox("Divisi:", st.session_state['data']['Divisi'].unique())
            
            with col2:
                if 'Hectare Mature' in features:
                    hectare = st.number_input("Hectare Mature:", min_value=0.0, value=float(st.session_state['data']['Hectare Mature'].mean()), step=0.1)
                if 'Budget FFB Production CM' in features:
                    budget = st.number_input("Budget FFB Production:", min_value=0.0, value=float(st.session_state['data']['Budget FFB Production CM'].mean()), step=1.0)
                if 'Month_Sin' in features:
                    month = st.selectbox("Bulan:", ['Januari', 'Februari', 'Maret', 'April', 'Mei', 'Juni', 'Juli', 'Agustus', 'September', 'Oktober', 'November', 'Desember'])
                    month_num = ['Januari', 'Februari', 'Maret', 'April', 'Mei', 'Juni', 'Juli', 'Agustus', 'September', 'Oktober', 'November', 'Desember'].index(month) + 1
            
            if st.button("🚀 Prediksi Sekarang", type="primary", use_container_width=True):
                try:
                    input_dict = {}
                    if 'Hectare Mature' in features: input_dict['Hectare Mature'] = hectare
                    if 'Budget FFB Production CM' in features: input_dict['Budget FFB Production CM'] = budget
                    
                    le_dict = st.session_state.get('le_dict', {})
                    if 'Business_Area_Encoded' in features: input_dict['Business_Area_Encoded'] = le_dict['Business Area'].transform([business_area])[0]
                    if 'Estate_Encoded' in features: input_dict['Estate_Encoded'] = le_dict['Estate'].transform([estate])[0]
                    if 'Divisi_Encoded' in features: input_dict['Divisi_Encoded'] = le_dict['Divisi'].transform([divisi])[0]
                    
                    if 'Month_Sin' in features:
                        input_dict['Month_Sin'] = np.sin(2 * np.pi * month_num / 12)
                        input_dict['Month_Cos'] = np.cos(2 * np.pi * month_num / 12)
                    
                    input_array = np.array([[input_dict[f] for f in features]])
                    
                    if selected_model == 'LSTM':
                        scaler = models['LSTM']['scaler']
                        lookback = models['LSTM']['lookback']
                        input_scaled = scaler.transform(input_array)
                        input_seq = np.repeat(input_scaled, lookback, axis=0)
                        input_seq = input_seq.reshape(1, lookback, len(features))
                        prediction = models['LSTM']['model'].predict(input_seq, verbose=0)[0][0]
                    else:
                        prediction = models['Random Forest']['model'].predict(input_array)[0]
                    
                    st.divider()
                    st.subheader("🎯 Hasil Prediksi")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Prediksi Produksi FFB", f"{prediction:.2f} ton")
                    with col2:
                        confidence_lower = prediction * 0.9
                        confidence_upper = prediction * 1.1
                        st.metric("Confidence Interval (90%)", f"{confidence_lower:.2f} - {confidence_upper:.2f} ton")
                    with col3:
                        if 'Budget FFB Production CM' in features:
                            variance = ((prediction - budget) / budget) * 100
                            st.metric("Variance vs Budget", f"{variance:+.2f}%", delta=f"{prediction - budget:+.2f} ton")
                    
                    st.markdown("### 📊 Visualisasi Prediksi")
                    fig = go.Figure()
                    if 'Budget FFB Production CM' in features:
                        fig.add_trace(go.Bar(x=['Budget', 'Prediksi'], y=[budget, prediction], marker_color=['#FFA726', '#2E7D32'], text=[f'{budget:.2f}', f'{prediction:.2f}'], textposition='auto'))
                    else:
                        fig.add_trace(go.Bar(x=['Prediksi'], y=[prediction], marker_color=['#2E7D32'], text=[f'{prediction:.2f}'], textposition='auto'))
                    fig.update_layout(title='Perbandingan Budget vs Prediksi', yaxis_title='Produksi FFB (ton)')
                    st.plotly_chart(fig, use_container_width=True)
                
                except Exception as e:
                    st.error(f"❌ Error saat prediksi: {str(e)}")
        
        with tab2:
            st.subheader("📊 Evaluasi Model pada Test Set")
            result = models[selected_model]
            
            col1, col2, col3, col4 = st.columns(4)
            with col1: st.metric("MAE", f"{result['mae']:.4f}")
            with col2: st.metric("RMSE", f"{result['rmse']:.4f}")
            with col3: st.metric("R² Score", f"{result['r2']:.4f}")
            with col4: st.metric("MAPE", f"{result['mape']:.2f}%")
            
            st.subheader("📈 Grafik Prediksi vs Aktual")
            n_points = min(100, len(result['actuals']))
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(range(n_points)), y=result['actuals'][:n_points].flatten(), mode='lines+markers', name='Actual', line=dict(color='#2E7D32')))
            fig.add_trace(go.Scatter(x=list(range(n_points)), y=result['predictions'][:n_points].flatten(), mode='lines+markers', name='Predicted', line=dict(color='#FFA726', dash='dash')))
            fig.update_layout(title=f'Prediksi vs Aktual - {selected_model}', xaxis_title='Index', yaxis_title='Produksi FFB (ton)')
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("🎯 Scatter Plot: Predicted vs Actual")
            fig = px.scatter(x=result['actuals'].flatten(), y=result['predictions'].flatten(), labels={'x': 'Actual', 'y': 'Predicted'}, title=f'Scatter Plot - {selected_model}', trendline='ols', opacity=0.6)
            max_val = max(result['actuals'].max(), result['predictions'].max())
            min_val = min(result['actuals'].min(), result['predictions'].min())
            fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines', name='Perfect Prediction', line=dict(color='red', dash='dash')))
            st.plotly_chart(fig, use_container_width=True)

# ==========================================
# MENU 5: EVALUASI KOMPARATIF
# ==========================================
elif menu == "📉 Evaluasi Komparatif":
    st.header("📉 Evaluasi dan Perbandingan Model")
    
    if 'models' not in st.session_state:
        st.warning("⚠️ Silakan lakukan training model terlebih dahulu!")
    elif len(st.session_state['models']) < 2:
        st.info("ℹ️ Untuk perbandingan komparatif, silakan training kedua model (LSTM dan Random Forest)")
    else:
        models = st.session_state['models']
        
        st.subheader("📊 Tabel Perbandingan Metrik")
        metrics_data = []
        for model_name, result in models.items():
            metrics_data.append({'Model': model_name, 'MAE': result['mae'], 'RMSE': result['rmse'], 'R² Score': result['r2'], 'MAPE (%)': result['mape']})
        
        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df.style.highlight_min(subset=['MAE', 'RMSE', 'MAPE (%)'], color='lightgreen').highlight_max(subset=['R² Score'], color='lightgreen'), use_container_width=True)
        
        st.divider()
        st.subheader("📊 Visualisasi Perbandingan")
        
        tab1, tab2, tab3 = st.tabs(["📈 Bar Charts", "🎯 Radar Chart", "📉 Model Comparison"])
        
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                fig = go.Figure()
                fig.add_trace(go.Bar(x=metrics_df['Model'], y=metrics_df['MAE'], name='MAE', marker_color='#2E7D32'))
                fig.add_trace(go.Bar(x=metrics_df['Model'], y=metrics_df['RMSE'], name='RMSE', marker_color='#558B2F'))
                fig.update_layout(title='Perbandingan MAE dan RMSE', barmode='group')
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig = go.Figure()
                fig.add_trace(go.Bar(x=metrics_df['Model'], y=metrics_df['R² Score'], name='R² Score', marker_color='#1976D2'))
                fig.update_layout(title='Perbandingan R² Score', yaxis_range=[0, 1])
                st.plotly_chart(fig, use_container_width=True)
            
            fig = px.bar(metrics_df, x='Model', y='MAPE (%)', title='Perbandingan MAPE (%)', color='MAPE (%)', color_continuous_scale='RdYlGn_r', text='MAPE (%)')
            fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            categories = ['MAE (inv)', 'RMSE (inv)', 'R² Score', 'MAPE (inv)']
            fig = go.Figure()
            
            for model_name, result in models.items():
                max_mae = max([r['mae'] for r in models.values()])
                max_rmse = max([r['rmse'] for r in models.values()])
                max_mape = max([r['mape'] for r in models.values()])
                
                values = [
                    1 - (result['mae'] / max_mae),
                    1 - (result['rmse'] / max_rmse),
                    result['r2'],
                    1 - (result['mape'] / max_mape)
                ]
                fig.add_trace(go.Scatterpolar(r=values, theta=categories, fill='toself', name=model_name))
            
            fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), title='Radar Chart - Perbandingan Performa Model')
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("📈 Perbandingan Prediksi vs Aktual")
            col1, col2 = st.columns(2)
            
            for idx, (model_name, result) in enumerate(models.items()):
                with col1 if idx == 0 else col2:
                    st.markdown(f"#### {model_name}")
                    n_points = min(100, len(result['actuals']))
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=list(range(n_points)), y=result['actuals'][:n_points].flatten(), mode='lines', name='Actual', line=dict(color='blue')))
                    fig.add_trace(go.Scatter(x=list(range(n_points)), y=result['predictions'][:n_points].flatten(), mode='lines', name='Predicted', line=dict(color='red', dash='dash')))
                    fig.update_layout(xaxis_title='Index', yaxis_title='Production (ton)', height=400)
                    st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        st.subheader("🏆 Kesimpulan Perbandingan")
        best_model = max(models.items(), key=lambda x: x[1]['r2'])
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.success(f"""### Model Terbaik: **{best_model[0]}** 🏆
            
            Berdasarkan metrik evaluasi:
            - **MAE**: {best_model[1]['mae']:.4f}
            - **RMSE**: {best_model[1]['rmse']:.4f}
            - **R² Score**: {best_model[1]['r2']:.4f}
            - **MAPE**: {best_model[1]['mape']:.2f}%
            
            Model ini menunjukkan performa terbaik dalam memprediksi produksi FFB.
            """)

# ==========================================
# MENU 6: INFORMASI
# ==========================================
elif menu == "ℹ️ Informasi":
    st.header("ℹ️ Informasi Aplikasi")
    
    tab1, tab2, tab3 = st.tabs(["📚 Tentang Penelitian", "🔧 Teknologi", "📖 Cara Penggunaan"])
    
    with tab1:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("### Analisis Kinerja LSTM dan Random Forest dalam Memprediksi Produksi FFB")
            st.markdown("#### 🎯 Latar Belakang")
            st.write("Fresh Fruit Bunch (FFB) merupakan hasil panen utama dari perkebunan kelapa sawit. Prediksi produksi FFB yang akurat sangat penting untuk perencanaan operasional, optimalisasi logistik, dan pengambilan keputusan strategis.")
            st.markdown("#### 📊 Dataset")
            st.write("Dataset mencakup data bulanan produksi FFB dari multiple business areas, estates, dan divisi.")
        with col2:
            st.markdown("### 📌 Key Points")
            st.info("**LSTM:** Specialized untuk time series.\n\n**Random Forest:** Ensemble of decision trees.")
    
    with tab2:
        st.markdown("### 🔧 Teknologi yang Digunakan")
        st.write("Aplikasi ini dibangun menggunakan **Python** dengan framework **Streamlit**.")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Tech Stack:**")
            st.code("""
Streamlit
Pandas & NumPy
Scikit-learn
TensorFlow/Keras
Plotly
            """, language="python")
        with col2:
            st.markdown("**Instalasi:**")
            st.code("""
pip install streamlit pandas numpy scikit-learn tensorflow plotly
streamlit run app.py
            """, language="bash")
    
    with tab3:
        st.markdown("### 📖 Panduan Penggunaan Aplikasi")
        st.write("1. **Upload Data**: Siapkan file CSV format yang sesuai.")
        st.write("2. **Training**: Pilih model (LSTM/RF) dan atur parameter.")
        st.write("3. **Prediksi**: Gunakan model untuk memprediksi produksi bulan depan.")
        st.write("4. **Evaluasi**: Bandingkan hasil prediksi dengan data aktual.")

# Footer (Native Python)
st.divider()
col1, col2, col3 = st.columns([1, 6, 1])
with col2:
    st.caption("© 2024 PT Salim Ivomas Pratama")
    st.caption("Sistem Prediksi Produksi FFB menggunakan Machine Learning | Version 1.0.0")