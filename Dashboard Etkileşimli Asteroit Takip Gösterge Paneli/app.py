import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# set_page_config must be the very first Streamlit call
st.set_page_config(
    page_title="Asteroit Takip Gosterge Paneli",
    page_icon="\u2604\ufe0f",
    layout="wide",
)

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "..", "Possible Asteroid Impacts with Earth NASA Sentry")
IMPACTS_CSV = os.path.join(DATA_DIR, "impacts.csv")
ORBITS_CSV  = os.path.join(DATA_DIR, "orbits.csv")

# Features used by the ML model (no leakage: Palermo/Torino excluded)
ML_FEATURES = [
    "Asteroid Velocity",
    "Asteroid Diameter (km)",
    "Asteroid Magnitude",
    "Possible Impacts",
    "Period Length",
]

@st.cache_data
def load_data():
    impacts = pd.read_csv(IMPACTS_CSV)
    orbits  = pd.read_csv(ORBITS_CSV)

    impacts["Maximum Torino Scale"] = (
        pd.to_numeric(impacts["Maximum Torino Scale"].replace("(*)", "0"), errors="coerce")
        .fillna(0)
    )
    impacts.dropna(subset=["Asteroid Velocity", "Asteroid Diameter (km)"], inplace=True)

    # impacts: provisional designations (e.g. "2006 WP1")
    # orbits:  numbered asteroids (e.g. "433 Eros") — different object populations.
    # Left-join on Object Name; suffixes=("", "_orb") keeps impacts columns authoritative.
    merged = pd.merge(impacts, orbits, on="Object Name", how="left", suffixes=("", "_orb"))
    if "Asteroid Magnitude_orb" in merged.columns:
        merged.drop(columns=["Asteroid Magnitude_orb"], inplace=True)

    merged["Period Length"] = merged["Period End"] - merged["Period Start"]
    return merged, orbits

@st.cache_resource
def train_model(df):
    train_df = df[ML_FEATURES + ["Cumulative Impact Probability"]].dropna()
    X = train_df[ML_FEATURES].values
    y = np.log10(train_df["Cumulative Impact Probability"].values)
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)),
    ])
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="r2")
    pipeline.fit(X, y)
    importances = pipeline.named_steps["rf"].feature_importances_
    return pipeline, cv_scores, importances

df_all, df_orbits = load_data()
model, cv_scores, importances = train_model(df_all)

st.title("\u2604\ufe0f Etkilesimli Asteroit Takip Gosterge Paneli")
st.markdown("**Kaynak:** NASA Sentry \u2014 Possible Asteroid Impacts with Earth")

with st.sidebar:
    st.header("\U0001f52d Filtreler")
    year_min = int(df_all["Period Start"].min())
    year_max = int(df_all["Period End"].max())
    year_range = st.slider("Yil Araligi", year_min, year_max, (year_min, year_max), step=1)
    diam_min = float(df_all["Asteroid Diameter (km)"].min())
    diam_max = float(df_all["Asteroid Diameter (km)"].max())
    diam_range = st.slider(
        "Asteroid Capi (km)",
        round(diam_min, 3), round(diam_max, 3),
        (round(diam_min, 3), round(diam_max, 3)),
        step=0.001, format="%.3f",
    )
    sort_by = st.selectbox(
        "Tehlikeli Cisimleri Sirala",
        ["Cumulative Impact Probability", "Maximum Torino Scale"],
    )

mask = (
    (df_all["Period Start"] >= year_range[0])
    & (df_all["Period End"]   <= year_range[1])
    & (df_all["Asteroid Diameter (km)"] >= diam_range[0])
    & (df_all["Asteroid Diameter (km)"] <= diam_range[1])
)
df = df_all[mask].copy()

tab1, tab2, tab3 = st.tabs([
    "\U0001f6a8 Tehlike Analizi",
    "\U0001f30d Yorunge Analizi",
    "\U0001f916 Tahmin Modeli",
])

# ── TAB 1: Danger Analysis ─────────────────────────────────────────────────────
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Filtrelenmis Asteroid", len(df))
    c2.metric("Maks. Torino Olcegi",
              int(df["Maximum Torino Scale"].max()) if not df.empty else "\u2014")
    c3.metric("Carpismo Olasiligi (maks)",
              f"{df['Cumulative Impact Probability'].max():.2e}" if not df.empty else "\u2014")
    c4.metric("Ort. Hiz (km/s)",
              f"{df['Asteroid Velocity'].mean():.2f}" if not df.empty else "\u2014")
    st.divider()

    st.subheader("\U0001f6a8 Tehlikeli Asteroitler")
    TABLE_COLS = [
        "Object Name", "Period Start", "Period End",
        "Cumulative Impact Probability", "Maximum Torino Scale",
        "Asteroid Diameter (km)", "Asteroid Velocity",
        "Cumulative Palermo Scale", "Maximum Palermo Scale",
    ]
    st.dataframe(
        df[TABLE_COLS].sort_values(sort_by, ascending=False).reset_index(drop=True),
        use_container_width=True, height=280,
    )
    st.divider()

    lc, rc = st.columns(2)
    with lc:
        st.subheader("\U0001f535 Hiz \u2013 Boyut Dagilimi")
        if df.empty:
            st.info("Filtreye uyan veri yok.")
        else:
            df["_sz"] = df["Maximum Torino Scale"] + 0.1
            fig_sc = px.scatter(
                df, x="Asteroid Velocity", y="Asteroid Diameter (km)",
                color="Maximum Torino Scale", size="_sz", size_max=28,
                hover_name="Object Name",
                hover_data={
                    "Cumulative Impact Probability": ":.2e",
                    "Period Start": True, "Period End": True, "_sz": False,
                },
                color_continuous_scale="YlOrRd",
                labels={
                    "Asteroid Velocity": "Hiz (km/s)",
                    "Asteroid Diameter (km)": "Cap (km)",
                    "Maximum Torino Scale": "Torino",
                },
                title="Asteroid Hizi ve Boyutu",
            )
            st.plotly_chart(fig_sc, use_container_width=True)

    with rc:
        st.subheader("\U0001f4ca En Tehlikeli 10 Asteroid")
        if df.empty:
            st.info("Filtreye uyan veri yok.")
        else:
            top10 = df.sort_values("Cumulative Impact Probability", ascending=False).head(10).copy()
            top10["Label"] = top10["Object Name"].str[:16]
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                name="Olasi Carpismo Sayisi", x=top10["Label"],
                y=top10["Possible Impacts"], marker_color="steelblue", yaxis="y1",
            ))
            fig_bar.add_trace(go.Scatter(
                name="Carpismo Olasiligi", x=top10["Label"],
                y=top10["Cumulative Impact Probability"],
                mode="lines+markers",
                marker=dict(size=8, color="tomato"),
                line=dict(color="tomato", width=2), yaxis="y2",
            ))
            fig_bar.update_layout(
                title="Olasi Carpismo Sayisi ve Kumulatif Olasilik (Top 10)",
                yaxis=dict(title="Olasi Carpismo Sayisi", side="left"),
                yaxis2=dict(title="Carpismo Olasiligi", overlaying="y",
                            side="right", showgrid=False),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                xaxis_tickangle=-35,
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    with st.expander("\U0001f4cb Tum Birlesik Veriyi Goster"):
        st.dataframe(df, use_container_width=True)

# ── TAB 2: Orbital Analysis ────────────────────────────────────────────────────
with tab2:
    st.info(
        "**Not:** `impacts.csv` gecici adlar (orn. '2006 WP1'), `orbits.csv` ise "
        "numarali asteroitler (orn. '433 Eros') icerdigi icin iki veri seti "
        "`Object Name` uzerinden eslesmiyor. Yorunge analizi `orbits.csv` "
        "uzerinden bagimsiz olarak yapilmaktadir."
    )
    oa1, oa2 = st.columns(2)
    with oa1:
        st.subheader("\U0001fa90 Siniflandirma Dagilimi")
        cls_df = df_orbits["Object Classification"].value_counts().reset_index()
        cls_df.columns = ["Sinif", "Sayi"]
        fig_pie = px.pie(cls_df, names="Sinif", values="Sayi",
                         title="Asteroid Turleri", hole=0.35)
        st.plotly_chart(fig_pie, use_container_width=True)
    with oa2:
        st.subheader("\U0001f4d0 Eksantriklik \u2013 Egim")
        orb_c = df_orbits.dropna(subset=["Orbit Eccentricity", "Orbit Inclination (deg)"])
        fig_ec = px.scatter(
            orb_c.sample(min(2000, len(orb_c)), random_state=1),
            x="Orbit Eccentricity", y="Orbit Inclination (deg)",
            color="Object Classification", opacity=0.6,
            labels={
                "Orbit Eccentricity": "Eksantriklik",
                "Orbit Inclination (deg)": "Egim (deg)",
                "Object Classification": "Tur",
            },
            title="Yorunge Eksantrikligi ve Egimi",
        )
        st.plotly_chart(fig_ec, use_container_width=True)

    st.subheader("\U0001f4cf Minimum Yorunge Kesisim Mesafesi (MOID)")
    moid = df_orbits["Minimum Orbit Intersection Distance (AU)"].dropna()
    fig_moid = px.histogram(moid, nbins=80, log_y=True,
                             labels={"value": "MOID (AU)", "count": "Sayi (log)"},
                             title="MOID Dagilimi \u2014 Dunyaya Yakin NEOlar")
    fig_moid.add_vline(x=0.05, line_dash="dash", line_color="red",
                       annotation_text="PHA Esigi 0.05 AU")
    st.plotly_chart(fig_moid, use_container_width=True)

# ── TAB 3: ML Prediction Model ────────────────────────────────────────────────
with tab3:
    st.subheader("\U0001f916 Carpismo Olasiligi Tahmin Modeli")
    st.markdown(
        "**Model:** Random Forest Regressor \u2014 log10(Cumulative Impact Probability) tahmini  \n"
        f"**Ozellikler:** {', '.join(ML_FEATURES)}  \n"
        f"**5-katli Capraz Dogrulama R\u00b2:** {cv_scores.mean():.3f} \u00b1 {cv_scores.std():.3f}"
    )

    mc1, mc2 = st.columns([1, 1])

    with mc1:
        st.subheader("\U0001f4c8 Ozellik Onem Puanlari")
        imp_df = pd.DataFrame({"Ozellik": ML_FEATURES, "Onem": importances})
        imp_df = imp_df.sort_values("Onem", ascending=True)
        fig_imp = px.bar(
            imp_df, x="Onem", y="Ozellik", orientation="h",
            color="Onem", color_continuous_scale="Blues",
            title="Random Forest \u2014 Ozellik Onem Siralamasi",
        )
        fig_imp.update_layout(showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig_imp, use_container_width=True)

    with mc2:
        st.subheader("\U0001f52e Interaktif Tahmin")
        st.markdown("Parametreleri ayarlayarak asteroidin tahmini carpismo olasiligini hesapla:")
        pred_vel  = st.slider("Asteroid Hizi (km/s)",
                              float(df_all["Asteroid Velocity"].min()),
                              float(df_all["Asteroid Velocity"].max()),
                              float(df_all["Asteroid Velocity"].median()), step=0.1)
        pred_diam = st.slider("Asteroid Capi (km)",
                              float(df_all["Asteroid Diameter (km)"].min()),
                              float(df_all["Asteroid Diameter (km)"].max()),
                              float(df_all["Asteroid Diameter (km)"].median()),
                              step=0.001, format="%.3f")
        pred_mag  = st.slider("Asteroid Parliakligi (Magnitude)",
                              float(df_all["Asteroid Magnitude"].min()),
                              float(df_all["Asteroid Magnitude"].max()),
                              float(df_all["Asteroid Magnitude"].median()), step=0.1)
        pred_pi   = st.slider("Olasi Carpismo Sayisi",
                              int(df_all["Possible Impacts"].min()),
                              int(df_all["Possible Impacts"].max()),
                              int(df_all["Possible Impacts"].median()), step=1)
        pred_pl   = st.slider("Gozlem Periyodu (yil)",
                              int(df_all["Period Length"].min()),
                              int(df_all["Period Length"].max()),
                              int(df_all["Period Length"].median()), step=1)

        X_pred    = np.array([[pred_vel, pred_diam, pred_mag, pred_pi, pred_pl]])
        log_prob  = model.predict(X_pred)[0]
        pred_prob = 10 ** log_prob

        st.divider()
        rc1, rc2 = st.columns(2)
        rc1.metric("Tahmini Carpismo Olasiligi", f"{pred_prob:.2e}")
        rc2.metric("log10(Olasilik)", f"{log_prob:.3f}")
        if pred_prob >= 1e-3:
            st.error("\U0001f534 YUKSEK RISK")
        elif pred_prob >= 1e-6:
            st.warning("\U0001f7e1 ORTA RISK")
        else:
            st.success("\U0001f7e2 DUSUK RISK")

    st.divider()
    st.subheader("\U0001f4ca Gercek vs Tahmin Edilen Degerler")
    tdf = df_all[ML_FEATURES + ["Cumulative Impact Probability", "Object Name"]].dropna()
    y_act = np.log10(tdf["Cumulative Impact Probability"].values)
    y_hat = model.predict(tdf[ML_FEATURES].values)
    avp_df = pd.DataFrame({
        "Gercek log10(CIP)": y_act,
        "Tahmin log10(CIP)": y_hat,
        "Object Name": tdf["Object Name"].values,
    })
    fig_avp = px.scatter(
        avp_df, x="Gercek log10(CIP)", y="Tahmin log10(CIP)",
        hover_name="Object Name", opacity=0.65,
        title="Model Dogrulugu: Gercek vs Tahmin",
    )
    lo = avp_df["Gercek log10(CIP)"].min()
    hi = avp_df["Gercek log10(CIP)"].max()
    fig_avp.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi], mode="lines",
        name="Mukemmel Tahmin", line=dict(dash="dash", color="red"),
    ))
    st.plotly_chart(fig_avp, use_container_width=True)

# streamlit run app.py