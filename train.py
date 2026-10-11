import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix, RocCurveDisplay)
import joblib
import os

os.makedirs("outputs", exist_ok=True)

# ======== 1. DOC & LAM SACH DU LIEU ========
df = pd.read_csv("data/CSDL_TieuDuong.csv")
print(f"Du lieu goc: {df.shape[0]} ban ghi")

df["GioiTinh"] = df["GioiTinh"].replace({
    "Male": "Nam", "Female": "Nu", "male": "Nam", "female": "Nu"
})

df["NguyCoTieuDuong"] = df["NguyCoTieuDuong"].replace({
    "TRUE": 1, "FALSE": 0, "True": 1, "False": 0,
    "Co nguy co": 1, "Khong co nguy co": 0,
    "Có nguy cơ": 1, "Không có nguy cơ": 0
})
df["NguyCoTieuDuong"] = pd.to_numeric(df["NguyCoTieuDuong"], errors="coerce")

so_cols = ["Tuoi", "Glucose", "HbA1C", "Cholesterol", "HDL",
           "LDL", "Triglycerid", "Creatinine", "Ure",
           "AcidUric", "AST", "ALT"]
for c in so_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

for col, max_val in [("Glucose", 50), ("HbA1C", 20), ("Cholesterol", 20)]:
    mask = df[col] > max_val
    while mask.any():
        df.loc[mask, col] = df.loc[mask, col] / 10
        mask = df[col] > max_val

df.loc[df["Creatinine"] < 10, "Creatinine"] = np.nan

hb_mode = df["HbA1C"].mode().iloc[0]
df["HbA1C_DoThuc"] = (df["HbA1C"] != hb_mode).astype(int)
df.loc[df["HbA1C_DoThuc"] == 0, "HbA1C"] = np.nan

df.drop(columns=["LDL"], inplace=True)
df = df.dropna(subset=["NguyCoTieuDuong"])
df["GioiTinh_Nam"] = (df["GioiTinh"] == "Nam").astype(int)

w_cols = ["Glucose", "HbA1C", "Cholesterol", "HDL",
          "Triglycerid", "Creatinine", "Ure",
          "AcidUric", "AST", "ALT", "Tuoi"]
for c in w_cols:
    q01 = df[c].quantile(0.01)
    q99 = df[c].quantile(0.99)
    df[c] = df[c].clip(lower=q01, upper=q99)

print(f"Du lieu sau lam sach: {df.shape}")

# ======== 2. DAC TRUNG & CHIA TAP ========
feature_cols = ["Tuoi", "GioiTinh_Nam", "Glucose", "HbA1C", "HbA1C_DoThuc",
                "Cholesterol", "HDL", "Triglycerid", "Creatinine",
                "Ure", "AcidUric", "AST", "ALT"]

X = df[feature_cols]
y = df["NguyCoTieuDuong"].astype(int)

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# ======== 3. HUEN LUYEN 5 MO HINH ========
model_configs = {
    "Logistic Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(C=0.1, class_weight="balanced",
                                     max_iter=1000, random_state=42))
    ]),
    "Random Forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                         random_state=42))
    ]),
    "SVM (RBF)": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", SVC(C=0.5, gamma=0.05, class_weight="balanced",
                      probability=True, random_state=42))
    ]),
    "KNN": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=5, weights="uniform"))
    ]),
    "Gradient Boosting": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", GradientBoostingClassifier(n_estimators=150, learning_rate=0.05,
                                             max_depth=2, random_state=42))
    ])
}

results = []
all_models = {}

for name, pipe in model_configs.items():
    print(f"Huen luyen {name}...")
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_val)
    y_proba = pipe.predict_proba(X_val)[:, 1]

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_val, y_pred),
        "Precision": precision_score(y_val, y_pred),
        "Recall": recall_score(y_val, y_pred),
        "F1": f1_score(y_val, y_pred),
        "ROC_AUC": roc_auc_score(y_val, y_proba)
    })
    all_models[name] = pipe

results_df = pd.DataFrame(results)
print("\nKet qua Validation:")
print(results_df.to_string(index=False))

# ======== 4. CHON MO HINH TOT NHAT ========
best_idx = results_df["F1"].idxmax()
best_name = results_df.loc[best_idx, "Model"]
best_pipe = all_models[best_name]
print(f"\nMo hinh tot nhat: {best_name}")

# Danh gia tren Test
y_pred_test = best_pipe.predict(X_test)
y_proba_test = best_pipe.predict_proba(X_test)[:, 1]

print(f"\nKet qua Test:")
print(f"  Accuracy:  {accuracy_score(y_test, y_pred_test):.4f}")
print(f"  Recall:    {recall_score(y_test, y_pred_test):.4f}")
print(f"  F1:        {f1_score(y_test, y_pred_test):.4f}")
print(f"  ROC-AUC:   {roc_auc_score(y_test, y_proba_test):.4f}")

cm = confusion_matrix(y_test, y_pred_test)
print(f"  TN={cm[0,0]}  FP={cm[0,1]}  FN={cm[1,0]}  TP={cm[1,1]}")

# ======== 5. TAO BIEU DO ========
print("\nDang tao bieu do...")

# --- Bieu do 1: Boxplot ngoai lai ---
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
fig.suptitle("Boxplot cac chi so (ngoai lai)", fontsize=14)
for i, c in enumerate(w_cols):
    if c in df.columns:
        ax = axes[i // 4][i % 4]
        ax.boxplot(df[c].dropna(), vert=True)
        ax.set_title(c)
plt.tight_layout()
plt.savefig("outputs/boxplot_ngoai_lai.png", dpi=150)
plt.close()
print("  -> outputs/boxplot_ngoai_lai.png")

# --- Bieu do 2: So sanh 5 mo hinh ---
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(results_df))
width = 0.15
colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"]
metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]
for i, m in enumerate(metrics):
    ax.bar(x + i * width, results_df[m], width, label=m, color=colors[i])
ax.set_xticks(x + width * 2)
ax.set_xticklabels(results_df["Model"], rotation=15)
ax.set_ylim(0, 1.1)
ax.set_title("So sanh 5 mo hinh (Validation)")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/so_sanh_mo_hinh.png", dpi=150)
plt.close()
print("  -> outputs/so_sanh_mo_hinh.png")

# --- Bieu do 3: Ma tran nham lan ---
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Khong nguy co", "Co nguy co"],
            yticklabels=["Khong nguy co", "Co nguy co"])
ax.set_title(f"Ma tran nham lan - {best_name}")
ax.set_xlabel("Du doan")
ax.set_ylabel("That")
plt.tight_layout()
plt.savefig("outputs/ma_tran_nham_lan.png", dpi=150)
plt.close()
print("  -> outputs/ma_tran_nham_lan.png")

# --- Bieu do 4: Duong cong ROC ---
fig, ax = plt.subplots(figsize=(8, 6))
for name, pipe in all_models.items():
    y_proba = pipe.predict_proba(X_test)[:, 1]
    RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax, name=name)
ax.set_title("Duong cong ROC (Test)")
plt.tight_layout()
plt.savefig("outputs/duong_cong_roc.png", dpi=150)
plt.close()
print("  -> outputs/duong_cong_roc.png")

# --- Bieu do 5: Do quan trong dac trung ---
if hasattr(best_pipe.named_steps["model"], "feature_importances_"):
    imp = best_pipe.named_steps["model"].feature_importances_
    fi = pd.DataFrame({"DacTrung": feature_cols, "DoQuanTrong": imp})
    fi = fi.sort_values("DoQuanTrong", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(fi["DacTrung"], fi["DoQuanTrong"], color="#2196F3")
    ax.set_title(f"Do quan trong dac trung - {best_name}")
    plt.tight_layout()
    plt.savefig("outputs/do_quan_trong.png", dpi=150)
    plt.close()
    print("  -> outputs/do_quan_trong.png")

# ======== 6. LUU MO HINH ========
joblib.dump(best_pipe, "outputs/mo_hinh_tot_nhat.joblib")
results_df.to_csv("outputs/ket_qua_danh_gia.csv", index=False)

# Luu log
with open("outputs/log_huan_luyen.txt", "w", encoding="utf-8") as f:
    f.write("=== LOG HUEN LUYEN ===\n")
    f.write(f"So ban ghi: {len(df)}\n")
    f.write(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}\n\n")
    f.write("Ket qua Validation:\n")
    f.write(results_df.to_string(index=False))
    f.write(f"\n\nMo hinh tot nhat: {best_name}\n")
    f.write(f"Test - Acc: {accuracy_score(y_test, y_pred_test):.4f}\n")
    f.write(f"Test - Rec: {recall_score(y_test, y_pred_test):.4f}\n")
    f.write(f"Test - F1:  {f1_score(y_test, y_pred_test):.4f}\n")
    f.write(f"Test - AUC: {roc_auc_score(y_test, y_proba_test):.4f}\n")

print("\nDA LUU TAT CA VAO outputs/!")
print("HOAN TAT!")