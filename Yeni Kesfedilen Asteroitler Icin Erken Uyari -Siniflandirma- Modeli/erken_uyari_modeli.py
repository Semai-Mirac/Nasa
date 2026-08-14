# =============================================================
# Yeni Kesfedilen Asteroitler - Erken Uyari Siniflandirma Modeli
# Random Forest & XGBoost ile PHA Tahmini
# =============================================================
# NOT: impacts.csv ve orbits.csv'deki Object Name formatlari
# birbirinden farkli (impacts: provisional desig., orbits: numbered/named).
# Bu nedenle model yalnizca orbits.csv verisini kullanir; hedef degisken
# MOID <= 0.05 AU kriteriyle olusturulur (NASA PHA tanimi).
# =============================================================

import os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix)
from xgboost import XGBClassifier

# ---------------------------------------------
# 0. DOSYA YOLLARI
# ---------------------------------------------
DATA_DIR = r"C:\Users\semai\OneDrive\Desktop\nasa protatip\Nasa\Possible Asteroid Impacts with Earth NASA Sentry"
OUT_DIR  = r"C:\Users\semai\OneDrive\Desktop\nasa protatip\Nasa\Yeni Kesfedilen Asteroitler Icin Erken Uyari -Siniflandirma- Modeli"

orbits_path = os.path.join(DATA_DIR, "orbits.csv")

# ---------------------------------------------
# 1. VERİ YÜKLEME
# ---------------------------------------------
print("=" * 65)
print("1. VERİ YÜKLENİYOR  (orbits.csv)")
df = pd.read_csv(orbits_path)
print(f"   Satir: {df.shape[0]}  Sutun: {df.shape[1]}")
print(f"   Sutunlar: {df.columns.tolist()}")

# ---------------------------------------------
# 2. HEDEF DEĞİŞKEN  (Is_Hazardous)
#    PHA kriteri: MOID <= 0.05 AU
# ---------------------------------------------
print("\n2. HEDEF DEGiSKEN OLUSTURULUYORe (Is_Hazardous) ...")

MOID_COL = "Minimum Orbit Intersection Distance (AU)"
df["Is_Hazardous"] = (df[MOID_COL].fillna(9999) <= 0.05).astype(int)

n_haz  = df["Is_Hazardous"].sum()
n_safe = (df["Is_Hazardous"] == 0).sum()
print(f"   Tehlikeli  (1) : {n_haz}")
print(f"   Tehlikesiz (0) : {n_safe}")
print(f"   Oran           : 1 tehlikeli / {n_safe/max(n_haz,1):.2f} tehlikesiz")

# ---------------------------------------------
# 3. ÖZELLİK MÜHENDİSLİĞİ
# ---------------------------------------------
print("\n3. OZELLIK MUHENDISLIGI...")

# Object Classification kategorik -> sayisal
le = LabelEncoder()
df["Classification_Code"] = le.fit_transform(df["Object Classification"].fillna("Unknown"))
print(f"   Sinif kategorileri: {list(enumerate(le.classes_))}")

# Apolyon mesafesi = Orbit Axis * (1 + Eccentricity)  -- dogrulama
# Tisserand parametresi tahmin edicisi (Jupiter yorbita icin yaklasisal)
df["Tisserand_approx"] = (5.2 / df["Orbit Axis (AU)"]) + 2 * np.sqrt(
    df["Orbit Axis (AU)"] / 5.2 * (1 - df["Orbit Eccentricity"]**2)
) * np.cos(np.radians(df["Orbit Inclination (deg)"]))

FEATURES = [
    "Orbit Axis (AU)",
    "Orbit Eccentricity",
    "Orbit Inclination (deg)",
    "Perihelion Distance (AU)",
    "Aphelion Distance (AU)",
    "Asteroid Magnitude",
    "Classification_Code",
    "Tisserand_approx",
]

# Leakage: MOID modele girmiyor
X = df[FEATURES].copy()
y = df["Is_Hazardous"].copy()

print(f"   X boyutu: {X.shape}")
missing = X.isnull().sum()
missing_cols = missing[missing > 0]
if len(missing_cols):
    print(f"   Eksik deger:\n{missing_cols.to_string()}")
else:
    print("   Eksik deger yok.")

# ---------------------------------------------
# 4. VERİ ÖN İŞLEME
# ---------------------------------------------
print("\n4. VERİ ON ISLEME...")

# Medyan ile eksik degerleri doldur
imputer = SimpleImputer(strategy="median")
X_imp = pd.DataFrame(imputer.fit_transform(X), columns=FEATURES)

# %80 Egitim / %20 Test (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X_imp, y, test_size=0.20, random_state=42, stratify=y
)
print(f"   Egitim : {X_train.shape[0]} ornek | Test : {X_test.shape[0]} ornek")

# StandardScaler
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
print("   StandardScaler uygulandı.")

# ---------------------------------------------
# 5. MODEL EĞİTİMİ
# ---------------------------------------------
print("\n5. MODELLER EGİTİLİYOR...")

n_neg    = (y_train == 0).sum()
n_pos    = (y_train == 1).sum()
scale_pw = round(n_neg / max(n_pos, 1), 3)
print(f"   Sinif orani -> Negatif: {n_neg}  Pozitif: {n_pos}  scale_pos_weight={scale_pw}")

rf = RandomForestClassifier(
    n_estimators=300, class_weight="balanced",
    random_state=42, n_jobs=-1
)
rf.fit(X_train_sc, y_train)
print("   [OK] Random Forest  (n_estimators=300, class_weight=balanced)")

xgb = XGBClassifier(
    n_estimators=300, learning_rate=0.05, max_depth=6,
    scale_pos_weight=scale_pw,
    eval_metric="logloss", random_state=42, n_jobs=-1, verbosity=0
)
xgb.fit(X_train_sc, y_train)
print(f"   [OK] XGBoost  (n_estimators=300, scale_pos_weight={scale_pw})")

# ---------------------------------------------
# 6. DEĞERLENDİRME
# ---------------------------------------------
print("\n" + "=" * 65)
print("6. MODEL DEGERLENDİRME RAPORU")
print("=" * 65)

results = {}

def evaluate_model(name, model, Xte, yte):
    y_pred = model.predict(Xte)
    acc  = accuracy_score(yte, y_pred)
    prec = precision_score(yte, y_pred, zero_division=0)
    rec  = recall_score(yte, y_pred, zero_division=0)
    f1   = f1_score(yte, y_pred, zero_division=0)
    print(f"\n  -- {name} --------------------------")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}  *** Tehlikeli asteroit kacirma riski! ***")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"\n  Siniflandirma Raporu:")
    print(classification_report(yte, y_pred,
                                target_names=["Tehlikesiz(0)", "Tehlikeli(1)"],
                                zero_division=0))
    results[name] = {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1-Score": f1}
    return y_pred

rf_preds  = evaluate_model("Random Forest", rf,  X_test_sc, y_test)
xgb_preds = evaluate_model("XGBoost",       xgb, X_test_sc, y_test)

print("\n  === OZET KARSILASTIRMA ===")
print(pd.DataFrame(results).T.to_string(float_format="{:.4f}".format))

# ---------------------------------------------
# 7. FEATURE IMPORTANCE
# ---------------------------------------------
print("\n7. FEATURE IMPORTANCE GRAFiKLERi ÇİZİLİYOR...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Ozellik Onem Siralaması\n(Tehlikeliligi Belirleyen Parametreler)",
             fontsize=13, fontweight="bold", y=1.01)

FEATURE_LABELS = [
    "Orbit Ekseni (AU)", "Yörünge Dışmerkezliği", "Yörünge Eğimi (°)",
    "Günberi Mesafesi (AU)", "Günöte Mesafesi (AU)",
    "Asteroit Parlaklığı", "Sınıf Kodu", "Tisserand Parametresi"
]

for ax, model, title, cmap in [
    (axes[0], rf,  "Random Forest", "YlOrRd"),
    (axes[1], xgb, "XGBoost",       "YlGn"),
]:
    imp_vals = pd.Series(model.feature_importances_, index=FEATURE_LABELS).sort_values()
    colors = plt.cm.get_cmap(cmap)(np.linspace(0.3, 0.85, len(imp_vals)))
    bars = ax.barh(imp_vals.index, imp_vals.values, color=colors, edgecolor="white", height=0.65)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Onem Skoru", fontsize=10)
    ax.set_xlim(0, imp_vals.max() * 1.22)
    for bar, val in zip(bars, imp_vals.values):
        ax.text(val + imp_vals.max() * 0.015,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8.5)

plt.tight_layout()
fi_path = os.path.join(OUT_DIR, "feature_importance.png")
plt.savefig(fi_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"   Kaydedildi: {fi_path}")

# ---------------------------------------------
# 8. KARIŞIKLIK MATRİSİ
# ---------------------------------------------
print("8. KARISIKLIK MATRiSLERi ÇİZİLİYOR...")

fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
fig2.suptitle("Karisiklik Matrisi (Test Seti)", fontsize=13, fontweight="bold")

for ax, preds, title in [
    (axes2[0], rf_preds,  "Random Forest"),
    (axes2[1], xgb_preds, "XGBoost"),
]:
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Tehlikesiz", "Tehlikeli"],
                yticklabels=["Tehlikesiz", "Tehlikeli"],
                linewidths=0.5, linecolor="white")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Tahmin Edilen", fontsize=10)
    ax.set_ylabel("Gercek Etiket", fontsize=10)
    # Kirmizi cerceve: False Negative (kacirilan tehlikeli asteroit)
    ax.add_patch(plt.Rectangle((1, 0), 1, 1, fill=False,
                                edgecolor="red", lw=2.5))
    ax.text(1.5, 0.5, "FN\n(Kacirilan\nTehlike)",
            ha="center", va="center", color="red", fontsize=8, fontweight="bold")

plt.tight_layout()
cm_path = os.path.join(OUT_DIR, "confusion_matrix.png")
plt.savefig(cm_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"   Kaydedildi: {cm_path}")

# ---------------------------------------------
# 9. METRİK KARŞILAŞTIRMA
# ---------------------------------------------
print("9. METRiK KARSILASTIRMA GRAFiGi ÇİZİLİYOR...")

metrics   = ["Accuracy", "Precision", "Recall", "F1-Score"]
rf_vals   = [results["Random Forest"][m] for m in metrics]
xgb_vals  = [results["XGBoost"][m]       for m in metrics]

x_pos = np.arange(len(metrics))
width = 0.35

fig3, ax3 = plt.subplots(figsize=(9, 5))
bars1 = ax3.bar(x_pos - width/2, rf_vals,  width, label="Random Forest",
                color="#2196F3", alpha=0.85, edgecolor="white")
bars2 = ax3.bar(x_pos + width/2, xgb_vals, width, label="XGBoost",
                color="#FF9800", alpha=0.85, edgecolor="white")

ax3.set_xticks(x_pos)
ax3.set_xticklabels(metrics, fontsize=11)
ax3.set_ylim(0, 1.15)
ax3.set_ylabel("Skor", fontsize=11)
ax3.set_title("Model Metrik Karsilastirmasi\n(Recall = Tehlikeli Asteroiti Yakalama Orani)",
              fontsize=12, fontweight="bold")
ax3.legend(fontsize=10)
ax3.axhline(y=0.90, color="red", linestyle="--", alpha=0.4, linewidth=1.2)
ax3.text(3.5, 0.91, "0.90", color="red", fontsize=9, alpha=0.7)

for bar in [*bars1, *bars2]:
    h = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width() / 2, h + 0.012,
             f"{h:.3f}", ha="center", va="bottom", fontsize=8.5)

plt.tight_layout()
mc_path = os.path.join(OUT_DIR, "metric_comparison.png")
plt.savefig(mc_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"   Kaydedildi: {mc_path}")

print("\n" + "=" * 65)
print("TUM ADIMLAR TAMAMLANDI.")
print(f"Ciktilar: {OUT_DIR}")
print("=" * 65)

