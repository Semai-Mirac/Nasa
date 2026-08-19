# %% [markdown]
# # Zaman Serisi Analizi ve Tehlike Takvimi (2026-2126)
#
# Bu proje, NASA Sentry "Possible Asteroid Impacts with Earth" veri setini (impacts.csv, orbits.csv)
# kullanarak gelecek 100 yil icin bir asteroid tehlike takvimi olusturur.
#
# ## Onemli veri notu
# impacts.csv icindeki tum satirlar zaten GELECEGE donuk risk pencereleridir (Period Start/End 2017-2880).
# orbits.csv ise 15000+ numarali/katalogli asteroidin statik yorunge elemanlarini icerir ve
# impacts.csv ile "Object Name" bazinda ortak kaydi YOKTUR (0 eslesme dogrulandi). Bu yuzden:
#  - orbits.csv, tamamlayici baglam (siniflandirma dagilimi, MOID) icin kullanilir, satir bazinda birlestirilmez.
#  - Gercek "zaman serisi tahmini" hedefi olarak, nesnelerin KESIF YILI (Object Name'deki yil onegi,
#    orn "2013 YB" -> 1979-2017 arasi) kullanilarak yillik "yeni risk nesnesi" serisi olusturulur ve
#    ADF/ARIMA ile 2026-2126 icin projekte edilir. Bu, bilinen risk pencerelerinin dogrudan
#    toplulastirilmasiyla birlestirilerek nihai "Tehlike Skoru" hesaplanir.
#
# ## Kurulum (VS Code entegre terminalinde calistirin)
# ```bash
# pip install pandas numpy matplotlib seaborn plotly statsmodels
# ```
# Prophet opsiyoneldir, kurulursa script otomatik kullanir:
# ```bash
# pip install prophet
# ```

# %%
# Adim 1: Kutuphaneler
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

DATA_DIR = r"C:\Users\semai\OneDrive\Desktop\nasa protatip\Nasa\Possible Asteroid Impacts with Earth NASA Sentry"
OUTPUT_DIR = r"C:\Users\semai\OneDrive\Desktop\nasa protatip\Nasa\Zaman Serisi Analizi ve Tehlike Takvimi"

impacts = pd.read_csv(DATA_DIR + r"\impacts.csv")
orbits = pd.read_csv(DATA_DIR + r"\orbits.csv")
print("impacts:", impacts.shape, "| orbits:", orbits.shape)
impacts.head()

# %%
# Adim 1: Tarih donusumu (pd.to_datetime) - Period End 2880 yilina kadar gidebildigi icin
# pandas Timestamp (ns) sinirini (yakl. yil 2262) asan satirlarda errors="coerce" ile NaT birakilir;
# hesaplamalarda kullanilan gercek yil bilgisi ayri int sutunlarda tutulur.
impacts["Period Start Year"] = impacts["Period Start"].astype(int)
impacts["Period End Year"] = impacts["Period End"].astype(int)
impacts["Period Start Date"] = pd.to_datetime(impacts["Period Start Year"].astype(str) + "-01-01", errors="coerce")
impacts["Period End Date"] = pd.to_datetime(impacts["Period End Year"].astype(str) + "-01-01", errors="coerce")

# Kesif yili: onegi "YYYY HARFLER" olan provizyonel adlardan, yoksa parantez icindeki YYYY'den
primary = impacts["Object Name"].str.extract(r"^(\d{4})\s+[A-Za-z]")[0]
fallback = impacts["Object Name"].str.extract(r"\((\d{4})\s")[0]
impacts["Discovery Year"] = primary.fillna(fallback).astype(int)
impacts.head()

# %%
# Orbits.csv - tamamlayici baglam (satir bazinda birlestirme yapilmaz, sadece genel istatistik)
print(orbits["Object Classification"].value_counts())
print("\nMOID (AU) ozet istatistik:")
print(orbits["Minimum Orbit Intersection Distance (AU)"].describe())

# %%
# Adim 1: Bilinen risk pencerelerini yillara dagitip kumulatif metrikler olusturma
def expand_to_years(row):
    years = np.arange(row["Period Start Year"], row["Period End Year"] + 1)
    n = len(years)
    return pd.DataFrame({
        "Year": years,
        "Object Name": row["Object Name"],
        "Impacts_Share": row["Possible Impacts"] / n,
        "Probability_Share": row["Cumulative Impact Probability"] / n,
        "Max_Palermo": row["Maximum Palermo Scale"],
        "Max_Torino": row["Maximum Torino Scale"],
    })

expanded = pd.concat([expand_to_years(r) for _, r in impacts.iterrows()], ignore_index=True)

yearly_known = expanded.groupby("Year").agg(
    Expected_Impacts=("Impacts_Share", "sum"),
    Cumulative_Probability=("Probability_Share", "sum"),
    Threatening_Objects=("Object Name", "nunique"),
    Max_Palermo=("Max_Palermo", "max"),
    Max_Torino=("Max_Torino", "max"),
).reset_index()
yearly_known.head()

# %%
# Yillik kesif serisi (gercek gecmis zaman serisi - ADF/ARIMA icin hedef)
discovery_counts = impacts.groupby("Discovery Year").size().rename("New_Risk_Objects")
full_range = pd.RangeIndex(discovery_counts.index.min(), discovery_counts.index.max() + 1, name="Discovery Year")
discovery_ts = discovery_counts.reindex(full_range, fill_value=0)
discovery_ts.plot(figsize=(11, 4), marker="o", title="Yillik Yeni Risk Listesi Nesnesi Sayisi (Kesif Yilina Gore)")
plt.tight_layout()
plt.show()

# %%
# Adim 2: Trend ve durgunluk (mevsimsellik yillik veride anlamli degil, 5 yillik hareketli ortalama ile trend)
fig, axes = plt.subplots(2, 1, figsize=(11, 7))
discovery_ts.plot(ax=axes[0], marker="o", label="Yillik sayi")
discovery_ts.rolling(5).mean().plot(ax=axes[0], color="red", label="5 yillik hareketli ortalama (trend)")
axes[0].set_title("Kesif Hizi Trendi")
axes[0].legend()
discovery_ts.diff().plot(ax=axes[1], color="darkorange", title="1. Fark (Durganlastirma)")
plt.tight_layout()
plt.show()

def run_adf(series, label):
    result = adfuller(series.dropna())
    print(f"--- ADF Testi: {label} ---")
    print(f"ADF istatistigi: {result[0]:.4f} | p-degeri: {result[1]:.4f}")
    print("Durgan" if result[1] < 0.05 else "Durgan degil")
    return result[1]

p_level = run_adf(discovery_ts, "Seviye (level)")
p_diff = run_adf(discovery_ts.diff(), "1. Fark")
d_order = 0 if p_level < 0.05 else 1

# %%
# Adim 3: 100 yillik projeksiyon icin ARIMA (kucuk/duzensiz yillik seri icin Prophet/LSTM'den daha uygun)
FORECAST_YEARS = list(range(2026, 2127))
n_forecast = len(FORECAST_YEARS)

best_aic, best_order = np.inf, (1, d_order, 1)
for p in range(0, 4):
    for q in range(0, 4):
        try:
            fitted = ARIMA(discovery_ts, order=(p, d_order, q)).fit()
            if fitted.aic < best_aic:
                best_aic, best_order = fitted.aic, (p, d_order, q)
        except Exception:
            continue
print("Secilen ARIMA order:", best_order, "| AIC:", round(best_aic, 2))

arima_model = ARIMA(discovery_ts, order=best_order).fit()
forecast_res = arima_model.get_forecast(steps=n_forecast)
forecast_mean = forecast_res.predicted_mean.clip(lower=0)
forecast_mean.index = FORECAST_YEARS

try:
    from prophet import Prophet
    prophet_df = discovery_ts.reset_index()
    prophet_df.columns = ["ds", "y"]
    prophet_df["ds"] = pd.to_datetime(prophet_df["ds"].astype(str) + "-01-01")
    prophet_model = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)
    prophet_model.fit(prophet_df)
    future = prophet_model.make_future_dataframe(periods=n_forecast, freq="YS")
    prophet_forecast = prophet_model.predict(future).set_index("ds")["yhat"].clip(lower=0)
    prophet_forecast.index = prophet_forecast.index.year
    forecast_mean = prophet_forecast.reindex(FORECAST_YEARS)
    print("Prophet bulundu, kesif hizi projeksiyonu Prophet ile hesaplandi.")
except ImportError:
    print("Prophet kurulu degil, ARIMA sonucu kullanilacak.")

# %%
# Adim 3 devam: Bilinen risk pencereleri + projekte edilen kesif hizinin birlesimi -> Tehlike Takvimi
future_known = yearly_known.set_index("Year").reindex(FORECAST_YEARS).fillna(0)

calendar = pd.DataFrame(index=pd.Index(FORECAST_YEARS, name="Year"))
calendar["Bilinen_Beklenen_Carpisma"] = future_known["Expected_Impacts"].values
calendar["Bilinen_Kumulatif_Olasilik"] = future_known["Cumulative_Probability"].values
calendar["Tehdit_Eden_Nesne_Sayisi"] = future_known["Threatening_Objects"].values
calendar["Projekte_Kesif_Hizi"] = forecast_mean.values

def normalize(s):
    span = s.max() - s.min()
    return (s - s.min()) / span if span > 0 else s * 0

calendar["Tehlike_Skoru"] = 100 * (
    0.45 * normalize(calendar["Bilinen_Kumulatif_Olasilik"]) +
    0.25 * normalize(calendar["Bilinen_Beklenen_Carpisma"]) +
    0.15 * normalize(calendar["Tehdit_Eden_Nesne_Sayisi"]) +
    0.15 * normalize(calendar["Projekte_Kesif_Hizi"])
)

top15 = calendar.sort_values("Tehlike_Skoru", ascending=False).head(15)
print("En riskli 15 yil:")
print(top15)

# %%
# Adim 4: Etkilesimli cizgi grafik (Plotly)
fig = go.Figure()
fig.add_trace(go.Scatter(x=calendar.index, y=calendar["Tehlike_Skoru"], mode="lines", name="Tehlike Skoru", line=dict(color="crimson", width=2)))
fig.add_trace(go.Scatter(x=top15.index, y=top15["Tehlike_Skoru"], mode="markers", name="En riskli 15 yil", marker=dict(color="black", size=9, symbol="star")))
fig.update_layout(title="2026-2126 Asteroid Tehlike Skoru Zaman Serisi", xaxis_title="Yil", yaxis_title="Tehlike Skoru (0-100)", template="plotly_white")
fig.write_html(OUTPUT_DIR + r"\tehlike_takvimi_cizgi_grafik.html")
fig.show()

# %%
# Adim 4: On yillik Isi Haritasi (Seaborn)
calendar["Decade"] = (calendar.index // 10) * 10
calendar["Year_in_Decade"] = calendar.index % 10
heat_data = calendar.pivot_table(index="Decade", columns="Year_in_Decade", values="Tehlike_Skoru")

plt.figure(figsize=(14, 8))
sns.heatmap(heat_data, annot=True, fmt=".0f", cmap="YlOrRd", cbar_kws={"label": "Tehlike Skoru"})
plt.title("2026-2126 On Yillik Tehlike Isi Haritasi")
plt.xlabel("On Yil Icindeki Yil")
plt.ylabel("On Yil Baslangici")
plt.tight_layout()
plt.savefig(OUTPUT_DIR + r"\tehlike_isi_haritasi.png", dpi=150)
plt.show()

# %%
# Sonuclarin disa aktarimi
export_cols = ["Bilinen_Beklenen_Carpisma", "Bilinen_Kumulatif_Olasilik", "Tehdit_Eden_Nesne_Sayisi", "Projekte_Kesif_Hizi", "Tehlike_Skoru"]
calendar[export_cols].to_csv(OUTPUT_DIR + r"\tehlike_takvimi_2026_2126.csv")
print("Kaydedildi:", OUTPUT_DIR)
print("\nEn yuksek riskli 10 yil ozeti:")
print(top15.head(10)[["Tehlike_Skoru", "Bilinen_Kumulatif_Olasilik", "Tehdit_Eden_Nesne_Sayisi"]])

