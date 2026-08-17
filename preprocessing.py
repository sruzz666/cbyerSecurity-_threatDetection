import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


columns = [
    "duration", "protocol_type", "service", "flag",
    "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
    "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count",
    "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate"
    , "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "attack_type", "difficulty_level"
]

train = pd.read_csv("KDDTrain+.txt", header=None, names=columns)
test = pd.read_csv("KDDTest+.txt", header=None, names=columns)


print("Missing values in train:", train.isnull().sum().sum())
print("Missing values in test:", test.isnull().sum().sum())
# NSL-KDD is typically clean, but if any exist:
train = train.fillna(train.median(numeric_only=True))
test = test.fillna(test.median(numeric_only=True))


attack_map = {
    'normal': 'normal',
    'back': 'dos', 'land': 'dos', 'neptune': 'dos', 'pod': 'dos',
    'smurf': 'dos', 'teardrop': 'dos', 'apache2': 'dos', 'udpstorm': 'dos',
    'processtable': 'dos', 'worm': 'dos', 'mailbomb': 'dos',
    'ipsweep': 'probe', 'nmap': 'probe', 'portsweep': 'probe',
    'satan': 'probe', 'mscan': 'probe', 'saint': 'probe',
    'ftp_write': 'r2l', 'guess_passwd': 'r2l', 'imap': 'r2l',
    'multihop': 'r2l', 'phf': 'r2l', 'spy': 'r2l', 'warezclient': 'r2l',
    'warezmaster': 'r2l', 'sendmail': 'r2l', 'named': 'r2l',
    'snmpgetattack': 'r2l', 'snmpguess': 'r2l', 'xlock': 'r2l',
    'xsnoop': 'r2l', 'httptunnel': 'r2l',
    'buffer_overflow': 'u2r', 'loadmodule': 'u2r', 'perl': 'u2r',
    'rootkit': 'u2r', 'ps': 'u2r', 'sqlattack': 'u2r', 'xterm': 'u2r'
}

train['attack_type'] = train['attack_type'].str.strip('.').map(attack_map)
test['attack_type'] = test['attack_type'].str.strip('.').map(attack_map)

# Drop any labels not covered by the map (rare, safety net)
train = train.dropna(subset=['attack_type'])
test = test.dropna(subset=['attack_type'])

print("\nTrain distribution:\n", train['attack_type'].value_counts(normalize=True))
print("\nTest distribution:\n", test['attack_type'].value_counts(normalize=True))

#  Encode categorical columns ─────────────────────────
categorical_cols = ['protocol_type', 'service', 'flag']


encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    combined = pd.concat([train[col], test[col]], axis=0)
    le.fit(combined)
    train[col] = le.transform(train[col])
    test[col] = le.transform(test[col])
    encoders[col] = le


label_encoder = LabelEncoder()
train['attack_type_encoded'] = label_encoder.fit_transform(train['attack_type'])
test['attack_type_encoded'] = label_encoder.transform(test['attack_type'])
print("\nClass mapping:", dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))

# ── 4. Drop difficulty_level (not a feature)
train = train.drop(columns=['difficulty_level'])
test = test.drop(columns=['difficulty_level'])

# ── 5. Separate features/labels ───────────────────────────
X_train_full = train.drop(columns=['attack_type', 'attack_type_encoded'])
y_train_full = train['attack_type_encoded']
X_test = test.drop(columns=['attack_type', 'attack_type_encoded'])
y_test = test['attack_type_encoded']

# ── 6. Scale numerical features ───────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_full)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train_full.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# ── 7. Train/validation split (from train set) ────────────
X_train, X_val, y_train, y_val = train_test_split(
    X_train_scaled, y_train_full,
    test_size=0.2, random_state=42, stratify=y_train_full
)

print(f"\nFinal shapes:")
print(f"X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test_scaled.shape}")

# ── 8. Handle class imbalance (class_weight approach — recommended) ──
# For tree-based models, class_weight='balanced' is simpler and safer
# than SMOTE on this kind of data (SMOTE can create unrealistic synthetic
# network traffic for extremely rare classes like u2r with only ~50 samples).
from sklearn.utils.class_weight import compute_class_weight

classes = np.unique(y_train)
weights = compute_class_weight('balanced', classes=classes, y=y_train)
class_weight_dict = dict(zip(classes, weights))
print("\nClass weights:", class_weight_dict)



from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_fscore_support
)
import matplotlib.pyplot as plt
import seaborn as sns

# ── 1. Train Random Forest with balanced class weights ────
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight=class_weight_dict,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

# ── 2. Evaluate on validation set ─────────────────────────
y_val_pred = rf.predict(X_val)

class_names = label_encoder.classes_  # ['dos','normal','probe','r2l','u2r']

print("=" * 60)
print("VALIDATION SET RESULTS")
print("=" * 60)
print(classification_report(y_val, y_val_pred, target_names=class_names, digits=4))

# ── 3. Evaluate on held-out TEST set (the real generalization test) ──
y_test_pred = rf.predict(X_test_scaled)

print("=" * 60)
print("TEST SET RESULTS (unseen attack variants included)")
print("=" * 60)
print(classification_report(y_test, y_test_pred, target_names=class_names, digits=4))

# ── 4. Confusion matrices ─────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

cm_val = confusion_matrix(y_val, y_val_pred)
sns.heatmap(cm_val, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names, ax=axes[0])
axes[0].set_title('Validation Set Confusion Matrix')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

cm_test = confusion_matrix(y_test, y_test_pred)
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Oranges',
            xticklabels=class_names, yticklabels=class_names, ax=axes[1])
axes[1].set_title('Test Set Confusion Matrix')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=150)
plt.show()

# ── 5. Feature importance ─────────────────────────────────
importances = pd.Series(rf.feature_importances_, index=X_train.columns)
importances_sorted = importances.sort_values(ascending=False)

print("\nTop 15 Most Important Features:")
print(importances_sorted.head(15))

plt.figure(figsize=(10, 8))
importances_sorted.head(20).plot(kind='barh')
plt.gca().invert_yaxis()
plt.title('Top 20 Feature Importances — Random Forest')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)
plt.show()

# ── 6. Save the trained model for later use (Streamlit app) ──
import joblib
joblib.dump(rf, 'rf_classifier.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')
joblib.dump(encoders, 'feature_encoders.pkl')
print("\nModels saved: rf_classifier.pkl, scaler.pkl, label_encoder.pkl, feature_encoders.pkl")



from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix

# ── 1. Isolate normal-only training data ──────────────────
normal_mask_train = (y_train == label_encoder.transform(['normal'])[0])
X_train_normal_only = X_train[normal_mask_train]

print(f"Training Isolation Forest on {X_train_normal_only.shape[0]} normal-only samples")

# ── 2. Train Isolation Forest ─────────────────────────────
iso_forest = IsolationForest(
    n_estimators=200,
    max_samples='auto',
    contamination=0.05,   # tune this — see notes below
    random_state=42,
    n_jobs=-1
)

iso_forest.fit(X_train_normal_only)

# ── 3. Predict on validation set ──────────────────────────
val_pred_raw = iso_forest.predict(X_val)
val_pred_binary = np.where(val_pred_raw == -1, 1, 0)  # 1 = anomaly, 0 = normal

y_val_binary = np.where(y_val == label_encoder.transform(['normal'])[0], 0, 1)

print("=" * 60)
print("ANOMALY DETECTOR — VALIDATION SET (binary: normal vs anomaly)")
print("=" * 60)
print(classification_report(y_val_binary, val_pred_binary,
                             target_names=['normal', 'anomaly'], digits=4))

# ── 4. Predict on test set ────────────────────────────────
test_pred_raw = iso_forest.predict(X_test_scaled)
test_pred_binary = np.where(test_pred_raw == -1, 1, 0)

y_test_binary = np.where(y_test == label_encoder.transform(['normal'])[0], 0, 1)

print("=" * 60)
print("ANOMALY DETECTOR — TEST SET (binary: normal vs anomaly)")
print("=" * 60)
print(classification_report(y_test_binary, test_pred_binary,
                             target_names=['normal', 'anomaly'], digits=4))

# ── 5. Does it catch what the classifier MISSED? (key result) ──
rf_missed_mask = (y_test != label_encoder.transform(['normal'])[0]) & \
                  (y_test_pred == label_encoder.transform(['normal'])[0])
print(f"\nAttacks the classifier misclassified as 'normal': {rf_missed_mask.sum()}")

caught_by_anomaly = test_pred_binary[rf_missed_mask.values] == 1
print(f"Of those, caught by anomaly detector: {caught_by_anomaly.sum()} "
      f"({100*caught_by_anomaly.mean():.1f}%)")

# ── 6. Anomaly score distribution ──────────────────────────
anomaly_scores = iso_forest.decision_function(X_test_scaled)

plt.figure(figsize=(10, 6))
for label in class_names:
    idx = y_test == label_encoder.transform([label])[0]
    plt.hist(anomaly_scores[idx], bins=50, alpha=0.5, label=label, density=True)
plt.xlabel('Anomaly Score (lower = more anomalous)')
plt.ylabel('Density')
plt.title('Anomaly Score Distribution by True Attack Type')
plt.legend()
plt.tight_layout()
plt.savefig('anomaly_scores.png', dpi=150)
plt.show()

# ── 7. Save the anomaly model ──────────────────────────────
joblib.dump(iso_forest, 'isolation_forest.pkl')
print("\nSaved: isolation_forest.pkl")
def hybrid_predict(X_data, rf_model, iso_model, label_encoder, class_names):
    """
    Combine Random Forest classifier + Isolation Forest anomaly detector
    into one final threat verdict per record.
    
    Logic:
      1. If RF confidently predicts a known attack type -> flag as that attack.
      2. If RF predicts 'normal' BUT anomaly detector disagrees -> flag as
         'unknown/suspicious' (this is the case that matters most).
      3. If both agree it's normal -> normal.
    """
    rf_pred = rf_model.predict(X_data)
    rf_proba = rf_model.predict_proba(X_data)
    rf_confidence = rf_proba.max(axis=1)

    anomaly_raw = iso_model.predict(X_data)
    is_anomaly = (anomaly_raw == -1)

    normal_class_idx = label_encoder.transform(['normal'])[0]

    results = []
    for i in range(len(X_data)):
        rf_label = class_names[rf_pred[i]]
        
        if rf_pred[i] != normal_class_idx:
            # RF thinks it's a known attack — trust it, note confidence
            verdict = rf_label
            reason = f"Classified as known {rf_label} attack (confidence: {rf_confidence[i]:.2f})"
        elif is_anomaly[i]:
            # RF says normal, but anomaly detector disagrees
            verdict = "unknown_suspicious"
            reason = "Predicted normal by classifier, but flagged as anomalous — possible unseen attack"
        else:
            # Both agree: normal
            verdict = "normal"
            reason = "Classified as normal, consistent with anomaly detector"

        results.append({"verdict": verdict, "reason": reason})

    return pd.DataFrame(results)

# ── Run hybrid system on test set ──────────────────────────
hybrid_results = hybrid_predict(X_test_scaled, rf, iso_forest, label_encoder, class_names)

print(hybrid_results['verdict'].value_counts())

# ── Evaluate: how many TRUE attacks fall into 'unknown_suspicious'? ──
true_labels_test = test['attack_type'].values  # original string labels
hybrid_results['true_label'] = true_labels_test

suspicious_mask = hybrid_results['verdict'] == 'unknown_suspicious'
print("\nTrue labels of records flagged 'unknown_suspicious':")
print(hybrid_results.loc[suspicious_mask, 'true_label'].value_counts())

# ── Overall hybrid detection rate (attack caught by EITHER layer) ──
is_actual_attack = (true_labels_test != 'normal')
is_flagged_by_hybrid = (hybrid_results['verdict'] != 'normal')

detected = is_actual_attack & is_flagged_by_hybrid
print(f"\nTotal actual attacks in test set: {is_actual_attack.sum()}")
print(f"Attacks flagged by hybrid system (either layer): {detected.sum()}")
print(f"Hybrid detection rate: {100*detected.sum()/is_actual_attack.sum():.2f}%")
print(f"(compare to RF-only detection rate: "
      f"{100*(y_test_pred != normal_class_idx).sum()/is_actual_attack.sum():.2f}%)" 
      if 'normal_class_idx' in dir() else "")

# ── Save hybrid results for the report ─────────────────────
hybrid_results.to_csv('hybrid_predictions.csv', index=False)
print("\nSaved: hybrid_predictions.csv")