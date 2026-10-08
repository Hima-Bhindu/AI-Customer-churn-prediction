"""
Churn Prediction & Behavior Intelligence Pipeline Module
Project: AI-Powered Customer Churn Intelligence Platform

This module provides the dynamic, dataset-independent ChurnPredictor class:
- Automatically detects Customer ID, Churn Target, and Activity features in any uploaded CSV.
- Preprocesses arbitrary numerical and categorical features dynamically (imputation, encoding, scaling).
- Evaluates model compatibility with pre-trained benchmark models or adaptively trains an ML classifier on the uploaded dataset.
- Derives customer-level Explainable AI risk factors based on feature contributions towards churn prediction.
- Performs unsupervised K-Means behavior mining on uploaded customer attributes, selecting optimal K via Silhouette Score.
- Projects behavior features into 2D via PCA for cluster visualization with exact N scatter points.
- Calculates supervised classification performance metrics (Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix, ROC Curve).
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score, accuracy_score, precision_score, 
    recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
)

class ChurnPredictor:
    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        
        # Load pre-trained benchmark model & scaling artifacts if present
        self.scaler = joblib.load(os.path.join(models_dir, "scaler.pkl")) if os.path.exists(os.path.join(models_dir, "scaler.pkl")) else None
        self.feature_columns = joblib.load(os.path.join(models_dir, "feature_columns.pkl")) if os.path.exists(os.path.join(models_dir, "feature_columns.pkl")) else None
        self.model = joblib.load(os.path.join(models_dir, "best_model.pkl")) if os.path.exists(os.path.join(models_dir, "best_model.pkl")) else None

    def detect_activity_column(self, df):
        """
        Detects active-day / activity column in the input DataFrame.
        Returns column_name or None. Does not invent fake activity data.
        """
        possible_cols = [
            'ActiveDays', 'Active_Days', 'DaysActive', 'UsageDays', 
            'LoginDays', 'ActivityDays', 'Days_Active', 'WatchDays', 'SessionDays'
        ]
        col_lower_map = {c.lower(): c for c in df.columns}
        
        for p in possible_cols:
            if p.lower() in col_lower_map:
                return col_lower_map[p.lower()]
        return None

    def detect_churn_column(self, df):
        """
        Detects historical churn target column if present.
        Returns column_name or None.
        """
        possible_targets = [
            'Churn', 'Churned', 'Exited', 'Cancelled', 'CustomerStatus', 
            'Status', 'Target', 'IsChurn', 'Churn_Label'
        ]
        col_lower_map = {c.lower(): c for c in df.columns}
        for t in possible_targets:
            if t.lower() in col_lower_map:
                return col_lower_map[t.lower()]
        return None

    def detect_id_column(self, df):
        """
        Detects customer ID column or returns None.
        """
        possible_ids = [
            'customerID', 'CustomerID', 'Customer_ID', 'ID', 'UserId', 
            'User_Id', 'AccountID', 'Client_ID', 'Subscriber_ID', 'User_ID'
        ]
        col_lower_map = {c.lower(): c for c in df.columns}
        for i in possible_ids:
            if i.lower() in col_lower_map:
                return col_lower_map[i.lower()]
        
        # Fallback to first object/string column if it ends with id or matches unique pattern
        for c in df.columns:
            if ('id' in c.lower() or 'key' in c.lower()) and df[c].dtype == 'object':
                return c
        return None

    def _generate_customer_explanations(self, df_raw, probas, feat_importances_df=None):
        """
        Generates Model Explanations ("Why Might This Customer Churn?") for each customer.
        Calculates feature risk contributions dynamically from the actual customer data.
        """
        explanations_list = []

        num_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()
        num_stats = {}
        for c in num_cols:
            std_val = df_raw[c].std()
            num_stats[c] = {
                'mean': df_raw[c].mean(),
                'std': std_val if std_val > 0 else 1.0
            }

        imp_map = {}
        if feat_importances_df is not None and not feat_importances_df.empty:
            imp_map = dict(zip(feat_importances_df['Feature'], feat_importances_df['Importance']))

        for idx, row in df_raw.iterrows():
            prob = probas[idx] if idx < len(probas) else 0.5
            factors = []

            # 1. Satisfaction / NPS (lower = risk)
            sat_col = None
            for col_candidate in ['SatisfactionScore', 'Satisfaction', 'NPS', 'Rating']:
                if col_candidate in df_raw.columns:
                    sat_col = col_candidate
                    break
            if sat_col and pd.notnull(row[sat_col]):
                val = float(row[sat_col])
                if val <= 2:
                    factors.append((0.9, f"Low {sat_col.lower()} ({int(val) if val.is_integer() else val:.1f}) contributed to the predicted churn risk"))

            # 2. Support Interactions / Complaints (higher = risk)
            supp_col = None
            for col_candidate in ['SupportCalls', 'SupportTickets', 'CustomerServiceCalls', 'Complaints']:
                if col_candidate in df_raw.columns:
                    supp_col = col_candidate
                    break
            if supp_col and pd.notnull(row[supp_col]):
                val = float(row[supp_col])
                if val >= 3:
                    factors.append((0.85, f"High {supp_col.lower()} ({int(val)} interactions) contributed to the predicted churn risk"))

            # 3. Active Days / Activity (lower = risk)
            act_col = self.detect_activity_column(df_raw)
            if act_col and pd.notnull(row[act_col]):
                val = float(row[act_col])
                mean_act = num_stats[act_col]['mean'] if act_col in num_stats else 15
                if val < max(5, mean_act * 0.4):
                    factors.append((0.8, f"Low active usage ({int(val)} days) contributed to the predicted churn risk"))

            # 4. Tenure (lower = risk)
            ten_col = None
            for col_candidate in ['tenure', 'TenureMonths', 'Tenure_Months', 'TenureInMonths']:
                if col_candidate in df_raw.columns:
                    ten_col = col_candidate
                    break
            if ten_col and pd.notnull(row[ten_col]):
                val = float(row[ten_col])
                if val <= 12:
                    factors.append((0.75, f"Short tenure ({int(val)} months) contributed to the predicted churn risk"))

            # 5. Financial Spend (higher spend relative to engagement = risk)
            spend_col = None
            for col_candidate in ['MonthlyCharges', 'MonthlySpend', 'Monthly_Spend', 'Monthly_Fee']:
                if col_candidate in df_raw.columns:
                    spend_col = col_candidate
                    break
            if spend_col and pd.notnull(row[spend_col]):
                val = float(row[spend_col])
                mean_spend = num_stats[spend_col]['mean'] if spend_col in num_stats else 60
                if val > mean_spend * 1.2:
                    factors.append((0.7, f"Higher monthly spend (${val:.2f}) contributed to the predicted churn risk"))

            # 6. Contract / Plan Type
            if 'Contract' in df_raw.columns and str(row['Contract']).lower() in ['month-to-month', 'monthly']:
                factors.append((0.65, "Month-to-month contract contributed to the predicted churn risk"))
            elif 'Plan' in df_raw.columns and str(row['Plan']).lower() in ['basic', 'free', 'starter']:
                factors.append((0.6, "Basic tier plan contributed to the predicted churn risk"))

            # 7. Payment Method / Auto-Pay
            if 'PaymentMethod' in df_raw.columns and 'check' in str(row['PaymentMethod']).lower():
                factors.append((0.55, "Manual check payment method contributed to the predicted churn risk"))

            # 8. Dynamic Feature Deviations for any remaining numerical features
            for nc in num_cols:
                if nc not in [sat_col, supp_col, act_col, ten_col, spend_col]:
                    val = float(row[nc]) if pd.notnull(row[nc]) else num_stats[nc]['mean']
                    z = (val - num_stats[nc]['mean']) / num_stats[nc]['std']
                    weight = imp_map.get(nc, 0.1)
                    if abs(z) > 1.2:
                        direction = "High" if z > 0 else "Low"
                        factors.append((abs(z) * weight, f"{direction} {nc} ({val:.1f}) contributed to the predicted churn risk"))

            factors.sort(key=lambda x: x[0], reverse=True)

            if prob >= 0.40 and len(factors) > 0:
                top_reasons = [f[1] for f in factors[:3]]
                explanations_list.append(" • ".join(top_reasons))
            elif prob < 0.40 and len(factors) > 0:
                top_reasons = [f[1] for f in factors[:2]]
                explanations_list.append(" • ".join(top_reasons))
            else:
                explanations_list.append("Stable behavioral retention profile contributed to lower churn risk")

        return explanations_list

    def predict_batch(self, df_input, high_thresh=0.60, low_thresh=0.30):
        """
        Dynamically processes ANY uploaded customer DataFrame:
        - Calculates customer-level churn probabilities, risk levels, behavior segments, and model explanations.
        - Selects optimal K for K-Means using Silhouette Score.
        - Projects behavior features into 2D via PCA for cluster visualization with exact N scatter points.
        - Computes classification performance metrics (Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix, ROC Curve).
        - Guarantees result row count EXACTLY equals uploaded dataset row count.
        """
        df_raw = df_input.copy()
        n_rows = len(df_raw)

        if n_rows == 0:
            return pd.DataFrame()

        # 1. Detect Customer ID
        id_col = self.detect_id_column(df_raw)
        if id_col:
            customer_ids = df_raw[id_col].astype(str)
        else:
            customer_ids = pd.Series([f"CUST-{i+1:04d}" for i in range(n_rows)], index=df_raw.index)

        # 2. Detect Activity Column
        act_col = self.detect_activity_column(df_raw)
        if act_col:
            active_days = pd.to_numeric(df_raw[act_col], errors='coerce').fillna(0)
            try:
                activity_groups = pd.qcut(
                    active_days, 
                    q=4, 
                    labels=['Bottom 25% (Low)', '25-50% (Moderate)', '50-75% (High)', 'Top 25% (Very High)'],
                    duplicates='drop'
                ).astype(str)
            except Exception:
                activity_groups = pd.Series(['Standard Activity'] * n_rows, index=df_raw.index)
        else:
            active_days = None
            activity_groups = pd.Series(['N/A'] * n_rows, index=df_raw.index)

        # 3. Detect Churn Target Column
        churn_col = self.detect_churn_column(df_raw)

        # Check pre-trained benchmark schema compatibility
        benchmark_cols = ['Contract', 'InternetService', 'PaymentMethod', 'MonthlyCharges', 'tenure']
        is_benchmark_schema = all(col in df_raw.columns for col in benchmark_cols)

        feat_importances_df = None
        is_adaptive = False
        model_status_msg = ""
        y_true = None

        if churn_col and churn_col in df_raw.columns:
            y_true = (df_raw[churn_col].astype(str).str.lower().isin(['yes', '1', 'true', 'churned'])).astype(int).values

        if is_benchmark_schema and self.model is not None and self.scaler is not None and self.feature_columns is not None:
            # Pre-trained Benchmark Model Inference
            try:
                df_bm = df_raw.copy()
                if 'tenure' not in df_bm.columns and 'TenureMonths' in df_bm.columns:
                    df_bm['tenure'] = df_bm['TenureMonths']
                if 'MonthlyCharges' not in df_bm.columns and 'MonthlySpend' in df_bm.columns:
                    df_bm['MonthlyCharges'] = df_bm['MonthlySpend']
                
                df_bm['tenure'] = pd.to_numeric(df_bm.get('tenure', 0), errors='coerce').fillna(0)
                df_bm['MonthlyCharges'] = pd.to_numeric(df_bm.get('MonthlyCharges', 0.0), errors='coerce').fillna(0.0)
                df_bm['TotalCharges'] = pd.to_numeric(df_bm.get('TotalCharges', df_bm['MonthlyCharges']*(df_bm['tenure']+1)), errors='coerce').fillna(0.0)
                
                service_cols = ['PhoneService', 'MultipleLines', 'InternetService', 
                                'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                                'TechSupport', 'StreamingTV', 'StreamingMovies']
                df_bm['TotalServices'] = 0
                for s in service_cols:
                    if s in df_bm.columns:
                        df_bm['TotalServices'] += (df_bm[s] == 'Yes').astype(int)

                df_bm['AvgMonthlyChargesPerTenure'] = df_bm['MonthlyCharges'] / (df_bm['tenure'] + 1)
                df_bm['TotalChargesPerTenure'] = df_bm['TotalCharges'] / (df_bm['tenure'] + 1)

                sec_col = df_bm['OnlineSecurity'] if 'OnlineSecurity' in df_bm.columns else pd.Series(['No']*n_rows)
                tech_col = df_bm['TechSupport'] if 'TechSupport' in df_bm.columns else pd.Series(['No']*n_rows)
                df_bm['SecuritySupportBundle'] = (((sec_col == 'Yes')) & ((tech_col == 'Yes'))).astype(int)

                tv_col = df_bm['StreamingTV'] if 'StreamingTV' in df_bm.columns else pd.Series(['No']*n_rows)
                mov_col = df_bm['StreamingMovies'] if 'StreamingMovies' in df_bm.columns else pd.Series(['No']*n_rows)
                df_bm['StreamingBundle'] = (((tv_col == 'Yes')) & ((mov_col == 'Yes'))).astype(int)

                df_bm['TenureGroup'] = pd.cut(df_bm['tenure'], bins=[-1, 12, 24, 48, 72], labels=['0-12m', '12-24m', '24-48m', '48-72m'])

                cols_to_drop = [c for c in [id_col, churn_col, 'Churn_Numeric', 'BehaviorSegment'] if c and c in df_bm.columns]
                X_raw = df_bm.drop(columns=cols_to_drop)

                cat_cols = X_raw.select_dtypes(include=['object', 'category']).columns.tolist()
                X_encoded = pd.get_dummies(X_raw, columns=cat_cols, drop_first=True)

                X_aligned = pd.DataFrame(0, index=X_encoded.index, columns=self.feature_columns)
                for col in X_encoded.columns:
                    if col in X_aligned.columns:
                        X_aligned[col] = X_encoded[col]

                X_scaled = pd.DataFrame(self.scaler.transform(X_aligned), columns=self.feature_columns, index=X_aligned.index)
                probas = self.model.predict_proba(X_scaled)[:, 1]
                model_status_msg = "Evaluated using pre-trained benchmark model"

                importances = self.model.feature_importances_
                feat_importances_df = pd.DataFrame({'Feature': self.feature_columns, 'Importance': importances}).sort_values('Importance', ascending=False)
            except Exception as e:
                is_benchmark_schema = False

        if not is_benchmark_schema:
            is_adaptive = True
            drop_cols = [c for c in [id_col, churn_col] if c and c in df_raw.columns]
            X_df = df_raw.drop(columns=drop_cols)

            # Variables used by the evaluation section below.
            acc = prec = rec = f1 = float('nan')
            auc_val = float('nan')
            cm = np.array([[0, 0], [0, 0]])
            roc_data = {
                'fpr': [0.0, 1.0],
                'tpr': [0.0, 1.0],
                'auc': None
            }

            if churn_col is not None:
                # ---------------------------------------------------------
                # ADAPTIVE MODEL
                # Train on 80% and evaluate on an unseen 20% test set.
                # ---------------------------------------------------------
                y_raw = df_raw[churn_col]

                y_num = (
                    y_raw.astype(str)
                    .str.strip()
                    .str.lower()
                    .isin(['yes', '1', 'true', 'churned', 'churn'])
                    .astype(int)
                    .values
                )

                cat_cols = X_df.select_dtypes(
                    include=['object', 'category']
                ).columns.tolist()

                num_cols = X_df.select_dtypes(
                    include=[np.number]
                ).columns.tolist()

                X_clean = X_df.copy()

                for nc in num_cols:
                    numeric_values = pd.to_numeric(
                        X_clean[nc],
                        errors='coerce'
                    )

                    median_value = (
                        numeric_values.median()
                        if not numeric_values.dropna().empty
                        else 0
                    )

                    X_clean[nc] = numeric_values.fillna(median_value)

                for cc in cat_cols:
                    X_clean[cc] = X_clean[cc].fillna('Unknown').astype(str)

                X_enc = pd.get_dummies(
                    X_clean,
                    columns=cat_cols,
                    drop_first=True
                )

                if not X_enc.empty and len(np.unique(y_num)) == 2:

                    scaler_adapt = StandardScaler()
                    X_scaled_adapt = scaler_adapt.fit_transform(X_enc)

                    # Stratified split keeps the churn/non-churn ratio similar.
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_scaled_adapt,
                        y_num,
                        test_size=0.20,
                        random_state=42,
                        stratify=y_num
                    )

                    adapt_clf = RandomForestClassifier(
                        n_estimators=100,
                        max_depth=6,
                        random_state=42,
                        class_weight='balanced'
                    )

                    # Train ONLY on the training portion.
                    adapt_clf.fit(X_train, y_train)

                    # -----------------------------------------------------
                    # TEST SET EVALUATION
                    # -----------------------------------------------------
                    test_probas = adapt_clf.predict_proba(X_test)[:, 1]
                    test_pred = (test_probas >= 0.50).astype(int)

                    acc = accuracy_score(y_test, test_pred)

                    prec = precision_score(
                        y_test,
                        test_pred,
                        zero_division=0
                    )

                    rec = recall_score(
                        y_test,
                        test_pred,
                        zero_division=0
                    )

                    f1 = f1_score(
                        y_test,
                        test_pred,
                        zero_division=0
                    )

                    cm = confusion_matrix(
                        y_test,
                        test_pred,
                        labels=[0, 1]
                    )

                    # ROC-AUC requires both classes in the test set.
                    if len(np.unique(y_test)) == 2:
                        auc_val = roc_auc_score(
                            y_test,
                            test_probas
                        )

                        fpr, tpr, _ = roc_curve(
                            y_test,
                            test_probas
                        )

                        roc_data = {
                            'fpr': fpr.tolist(),
                            'tpr': tpr.tolist(),
                            'auc': float(auc_val)
                        }

                    # -----------------------------------------------------
                    # PREDICT ALL UPLOADED CUSTOMERS
                    # These predictions are for the dashboard.
                    # Metrics above remain test-set metrics.
                    # -----------------------------------------------------
                    probas = adapt_clf.predict_proba(
                        X_scaled_adapt
                    )[:, 1]

                    imp = adapt_clf.feature_importances_

                    feat_importances_df = pd.DataFrame({
                        'Feature': X_enc.columns,
                        'Importance': imp
                    }).sort_values(
                        'Importance',
                        ascending=False
                    )

                    model_status_msg = (
                        f"Trained dynamic Random Forest using "
                        f"80% training / 20% testing split "
                        f"({n_rows} rows)"
                    )

                elif not X_enc.empty:
                    # A supervised model cannot be evaluated when the
                    # target contains only one class.
                    scaler_adapt = StandardScaler()
                    X_scaled_adapt = scaler_adapt.fit_transform(X_enc)

                    adapt_clf = RandomForestClassifier(
                        n_estimators=100,
                        max_depth=6,
                        random_state=42
                    )

                    adapt_clf.fit(X_scaled_adapt, y_num)

                    probas = adapt_clf.predict_proba(
                        X_scaled_adapt
                    )[:, 1]

                    imp = adapt_clf.feature_importances_

                    feat_importances_df = pd.DataFrame({
                        'Feature': X_enc.columns,
                        'Importance': imp
                    }).sort_values(
                        'Importance',
                        ascending=False
                    )

                    model_status_msg = (
                        "Only one churn class is present; "
                        "test-set classification metrics are unavailable"
                    )

                else:
                    probas = np.full(n_rows, 0.5)

                    model_status_msg = (
                        "Insufficient feature variation for ML training"
                    )

            else:
                # No historical churn target -> behavioral risk scoring.
                cat_cols = X_df.select_dtypes(
                    include=['object', 'category']
                ).columns.tolist()

                num_cols = X_df.select_dtypes(
                    include=[np.number]
                ).columns.tolist()

                X_clean = X_df.copy()

                for nc in num_cols:
                    numeric_values = pd.to_numeric(
                        X_clean[nc],
                        errors='coerce'
                    )

                    median_value = (
                        numeric_values.median()
                        if not numeric_values.dropna().empty
                        else 0
                    )

                    X_clean[nc] = numeric_values.fillna(median_value)

                for cc in cat_cols:
                    X_clean[cc] = X_clean[cc].fillna('Unknown').astype(str)

                X_enc = pd.get_dummies(
                    X_clean,
                    columns=cat_cols,
                    drop_first=True
                )

                if not X_enc.empty and len(num_cols) > 0:
                    scaler_adapt = StandardScaler()
                    X_scaled_adapt = scaler_adapt.fit_transform(X_enc)

                    risk_score = np.zeros(n_rows)
                    total_weight = 0.0

                    for c in num_cols:
                        std_value = X_clean[c].std()

                        z = (
                            X_clean[c] - X_clean[c].mean()
                        ) / (
                            std_value if std_value > 0 else 1.0
                        )

                        col_l = c.lower()

                        if (
                            'sat' in col_l
                            or 'nps' in col_l
                            or 'rating' in col_l
                            or 'active' in col_l
                            or 'usage' in col_l
                            or 'tenure' in col_l
                        ):
                            risk_score -= z * 1.5
                            total_weight += 1.5

                        elif (
                            'call' in col_l
                            or 'ticket' in col_l
                            or 'complaint' in col_l
                            or 'spend' in col_l
                            or 'charge' in col_l
                        ):
                            risk_score += z * 1.5
                            total_weight += 1.5

                        else:
                            risk_score += np.abs(z) * 0.5
                            total_weight += 0.5

                    if total_weight > 0:
                        risk_score = risk_score / total_weight

                    probas = 1 / (1 + np.exp(-risk_score))

                    var_imp = X_enc.var().values
                    total_var = (
                        var_imp.sum()
                        if var_imp.sum() > 0
                        else 1.0
                    )

                    feat_importances_df = pd.DataFrame({
                        'Feature': X_enc.columns,
                        'Importance': var_imp / total_var
                    }).sort_values(
                        'Importance',
                        ascending=False
                    )

                else:
                    probas = np.full(n_rows, 0.35)

                model_status_msg = (
                    "Historical churn target not found. "
                    "Evaluated using dynamic behavioral risk scoring."
                )

        # 4. Unsupervised K-Means Behavior Mining & Silhouette Score Selection
        drop_cols_km = [c for c in [id_col, churn_col] if c and c in df_raw.columns]
        X_km_df = df_raw.drop(columns=drop_cols_km)
        cat_km = X_km_df.select_dtypes(include=['object', 'category']).columns.tolist()
        num_km = X_km_df.select_dtypes(include=[np.number]).columns.tolist()

        X_km_clean = X_km_df.copy()
        for nc in num_km:
            X_km_clean[nc] = pd.to_numeric(X_km_clean[nc], errors='coerce').fillna(X_km_clean[nc].median() if not X_km_clean[nc].dropna().empty else 0)
        for cc in cat_km:
            X_km_clean[cc] = X_km_clean[cc].astype(str).fillna('Unknown')

        X_km_enc = pd.get_dummies(X_km_clean, columns=cat_km, drop_first=True)

        optimal_k = 3
        best_silhouette = 0.0
        pca_df = None
        cluster_labels_list = []
        segment_names = []

        if not X_km_enc.empty and n_rows >= 3:
            scaler_km = StandardScaler()
            X_km_scaled = scaler_km.fit_transform(X_km_enc)

            # Evaluate candidate K values using Silhouette Score
            max_k = min(6, n_rows - 1)
            if max_k >= 2:
                best_k = 2
                best_sil = -1.0
                for k_cand in range(2, max_k + 1):
                    try:
                        km_cand = KMeans(n_clusters=k_cand, random_state=42, n_init=10)
                        lbls_cand = km_cand.fit_predict(X_km_scaled)
                        if len(set(lbls_cand)) > 1:
                            sil = silhouette_score(X_km_scaled, lbls_cand)
                            if sil > best_sil:
                                best_sil = sil
                                best_k = k_cand
                    except Exception:
                        pass
                optimal_k = best_k
                best_silhouette = max(0.0, float(best_sil))
            else:
                optimal_k = 2
                best_silhouette = 0.50

            km_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
            cluster_ids = km_final.fit_predict(X_km_scaled)
            cluster_labels_list = [f"Cluster {c}" for c in cluster_ids]

            # 2D PCA projection for Behavior Mining Visualization
            if X_km_scaled.shape[1] >= 2:
                pca = PCA(n_components=2, random_state=42)
                pca_coords = pca.fit_transform(X_km_scaled)
                pca_df = pd.DataFrame({
                    'PCA_1': pca_coords[:, 0],
                    'PCA_2': pca_coords[:, 1],
                    'Cluster': cluster_labels_list,
                    'customerID': customer_ids
                }, index=df_raw.index)

            # Calculate cluster profile means to generate dynamic descriptive cluster names
            cluster_df = pd.DataFrame(X_km_clean[num_km] if len(num_km)>0 else X_km_scaled)
            cluster_df['Cluster_ID'] = cluster_ids
            means = cluster_df.groupby('Cluster_ID').mean()

            for cid in cluster_ids:
                if len(num_km) > 0:
                    c_row = means.loc[cid]
                    top_feat = c_row.idxmax()
                    val = c_row[top_feat]
                    segment_names.append(f"Cluster {cid}: High {top_feat} ({val:.1f})")
                else:
                    segment_names.append(f"Cluster {cid}: Behavioral Segment")
        else:
            optimal_k = 1
            best_silhouette = 1.00
            cluster_labels_list = ["Cluster 0"] * n_rows
            segment_names = ["Cluster 0: Core Customer Segment"] * n_rows

        # 5. Risk Classification
        risk_levels = []
        pred_labels = []
        for p in probas:
            pred_labels.append("Yes" if p >= 0.50 else "No")
            if p >= high_thresh:
                risk_levels.append("High Risk")
            elif p >= low_thresh:
                risk_levels.append("Medium Risk")
            else:
                risk_levels.append("Low Risk")

        # 6. Generate Model Explanations ("Why Might This Customer Churn?")
        explanations = self._generate_customer_explanations(df_raw, probas, feat_importances_df)

        # 7. Supervised Model Classification Performance Metrics
        #
        # IMPORTANT:
        # For the adaptive model, acc/prec/rec/f1/auc_val/cm/roc_data
        # were calculated above using the UNSEEN TEST SET.
        #
        # Never use y_pred = y_true as a fallback. That would produce
        # artificially perfect 100% metrics.

        if churn_col is not None and not is_adaptive:
            # Pre-trained benchmark model:
            # evaluate predictions against the uploaded historical labels.
            y_pred = (probas >= 0.50).astype(int)

            acc = accuracy_score(y_true, y_pred)

            prec = precision_score(
                y_true,
                y_pred,
                zero_division=0
            )

            rec = recall_score(
                y_true,
                y_pred,
                zero_division=0
            )

            f1 = f1_score(
                y_true,
                y_pred,
                zero_division=0
            )

            cm = confusion_matrix(
                y_true,
                y_pred,
                labels=[0, 1]
            )

            if len(np.unique(y_true)) == 2:
                auc_val = roc_auc_score(
                    y_true,
                    probas
                )

                fpr, tpr, _ = roc_curve(
                    y_true,
                    probas
                )

                roc_data = {
                    'fpr': fpr.tolist(),
                    'tpr': tpr.tolist(),
                    'auc': float(auc_val)
                }

        if churn_col is None:
            # No historical labels -> classification metrics are not valid.
            acc = prec = rec = f1 = float('nan')
            auc_val = float('nan')

            cm = np.array([
                [0, 0],
                [0, 0]
            ])

            roc_data = {
                'fpr': [0.0, 1.0],
                'tpr': [0.0, 1.0],
                'auc': None
            }

        metrics_dict = {
            'Accuracy': (
                f"{acc * 100:.2f}%"
                if not np.isnan(acc)
                else "N/A"
            ),
            'Precision': (
                f"{prec * 100:.2f}%"
                if not np.isnan(prec)
                else "N/A"
            ),
            'Recall': (
                f"{rec * 100:.2f}%"
                if not np.isnan(rec)
                else "N/A"
            ),
            'F1-Score': (
                f"{f1:.4f}"
                if not np.isnan(f1)
                else "N/A"
            ),
            'ROC-AUC': (
                f"{auc_val:.4f}"
                if not np.isnan(auc_val)
                else "N/A"
            )
        }

        # Assemble Output DataFrame (Guaranteed exact row count = n_rows)
        res_df = pd.DataFrame({
            'customerID': customer_ids,
            'ActiveDays': active_days if active_days is not None else np.nan,
            'ActivityGroup': activity_groups,
            'ChurnProbability_Raw': probas,
            'ChurnProbability': [f"{p*100:.1f}%" for p in probas],
            'PredictedChurn': pred_labels,
            'RiskLevel': risk_levels,
            'BehaviorCluster': cluster_labels_list,
            'BehaviorSegment': segment_names,
            'ImportantRiskFactors': explanations
        }, index=df_raw.index)

        res_df.attrs['is_adaptive'] = is_adaptive
        res_df.attrs['feat_importances'] = feat_importances_df
        res_df.attrs['model_status_msg'] = model_status_msg
        res_df.attrs['churn_target_present'] = churn_col is not None
        res_df.attrs['optimal_k'] = optimal_k
        res_df.attrs['best_silhouette'] = best_silhouette
        res_df.attrs['pca_df'] = pca_df
        res_df.attrs['metrics'] = metrics_dict
        res_df.attrs['confusion_matrix'] = cm
        res_df.attrs['roc_curve'] = roc_data

        return res_df

if __name__ == "__main__":
    predictor = ChurnPredictor()
    test_df = pd.DataFrame({
        "CustomerID": ["C001", "C002", "C003", "C004", "C005"],
        "TenureMonths": [2, 18, 45, 1, 60],
        "MonthlySpend": [95.0, 45.0, 70.0, 110.0, 25.0],
        "SupportCalls": [5, 1, 0, 6, 0],
        "SatisfactionScore": [1, 4, 5, 1, 5]
    })
    res = predictor.predict_batch(test_df)
    print("Execution Success. Customer count:", len(res))
    print(res[['customerID', 'ChurnProbability', 'RiskLevel', 'BehaviorCluster', 'ImportantRiskFactors']])
