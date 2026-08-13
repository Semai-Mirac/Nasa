# açmak için http://localhost:8501

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
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Asteroit Takip Paneli",
    page_icon="☄️",
    layout="wide",
)

# -- Meteor cursor: cartoon sparkles + flickering flame ------------------
st.markdown("<style>html,body,*{cursor:none!important;}</style>", unsafe_allow_html=True)
components.html("""<script>
(function(){
  var doc=window.parent.document,win=window.parent;
  var old=doc.getElementById('m-cvs');if(old)old.remove();
  var cvs=doc.createElement('canvas');
  cvs.id='m-cvs';
  cvs.style.cssText='position:fixed;top:0;left:0;pointer-events:none;z-index:2147483647;';
  doc.body.appendChild(cvs);
  var ctx=cvs.getContext('2d');
  function resize(){cvs.width=win.innerWidth;cvs.height=win.innerHeight;}
  resize();win.addEventListener('resize',resize);
  var tx=win.innerWidth/2,ty=win.innerHeight/2,mx=tx,my=ty;
  var hist=[],HIST=42,t=0,sparks=[];
  doc.addEventListener('mousemove',function(e){tx=e.clientX;ty=e.clientY;});

  function Spark(cx,cy,big){
    var a=Math.random()*6.2832,d=big?(5+Math.random()*20):(8+Math.random()*36);
    var pals=[[255,240,50],[255,200,20],[255,255,255],[255,155,0],[160,220,255],[255,100,200]];
    var p=pals[0|Math.random()*pals.length];
    this.x=cx+Math.cos(a)*d; this.y=cy+Math.sin(a)*d;
    this.vx=(Math.random()-0.5)*1.4; this.vy=(Math.random()-0.5)*1.4-0.4;
    this.sz=0; this.maxSz=big?(5+Math.random()*7):(4+Math.random()*6);
    this.life=1; this.decay=big?(0.007+Math.random()*0.009):(0.014+Math.random()*0.018);
    this.grow=true; this.r=p[0]; this.g=p[1]; this.b=p[2]; this.rot=Math.random()*6.28;
  }
  Spark.prototype.tick=function(){
    if(this.grow){this.sz+=0.7;if(this.sz>=this.maxSz)this.grow=false;}
    else{this.life-=this.decay;this.sz*=0.95;}
    this.x+=this.vx; this.y+=this.vy; this.rot+=0.06;
  };
  Spark.prototype.draw=function(c){
    if(this.life<=0||this.sz<0.1)return;
    var a=Math.min(1,this.life*this.life),s=this.sz;
    c.save();
    c.globalAlpha=a;
    c.translate(this.x,this.y); c.rotate(this.rot);
    c.shadowBlur=s*10; c.shadowColor='rgb('+this.r+','+this.g+','+this.b+')';
    c.fillStyle='rgb('+this.r+','+this.g+','+this.b+')';
    c.beginPath();
    c.moveTo(0,-s*4);
    c.quadraticCurveTo(s*0.9,-s*0.9, s*4,0);
    c.quadraticCurveTo(s*0.9,s*0.9,  0,s*4);
    c.quadraticCurveTo(-s*0.9,s*0.9, -s*4,0);
    c.quadraticCurveTo(-s*0.9,-s*0.9, 0,-s*4);
    c.closePath(); c.fill();
    c.shadowBlur=s*3; c.fillStyle='rgba(255,255,255,0.95)';
    c.beginPath(); c.arc(0,0,s*0.55,0,6.2832); c.fill();
    c.restore();
  };

  function drawFlame(c,h,cx,cy,tick){
    if(h.length<4)return;
    var tail=h[0],ddx=cx-tail.x,ddy=cy-tail.y,len=Math.sqrt(ddx*ddx+ddy*ddy)||1;
    var px=-ddy/len,py=ddx/len;
    var flk=1+0.12*Math.sin(tick*0.29)*Math.sin(tick*0.18+1.2);
    var W=28*flk;

    // Soft radial glow blobs along trail (no hard triangular edge)
    for(var i=4;i<h.length;i+=3){
      var tt=i/(h.length-1),r=W*1.8*tt*tt,alpha=0.07*tt*tt;
      if(r<1)continue;
      c.save();c.globalAlpha=alpha;
      var rg=c.createRadialGradient(h[i].x,h[i].y,0,h[i].x,h[i].y,r);
      rg.addColorStop(0,'rgba(255,200,30,1)');rg.addColorStop(1,'rgba(255,80,0,0)');
      c.fillStyle=rg;c.beginPath();c.arc(h[i].x,h[i].y,r,0,6.2832);c.fill();
      c.restore();
    }

    // Main wavy flame — long fade-in so tail dissolves smoothly
    c.save();
    c.beginPath();c.moveTo(tail.x,tail.y);
    for(var i=1;i<h.length;i++){
      var tt=i/(h.length-1),w=W*tt*tt;
      var wv=Math.sin(tick*0.27+i*0.58)*2.8*tt;
      c.lineTo(h[i].x+px*(w+wv),h[i].y+py*(w+wv));
    }
    for(var i=h.length-2;i>=1;i--){
      var tt=i/(h.length-1),w=W*tt*tt;
      var wv=Math.sin(tick*0.27+i*0.58+3.14)*2.8*tt;
      c.lineTo(h[i].x-px*(w+wv),h[i].y-py*(w+wv));
    }
    c.closePath();
    var g1=c.createLinearGradient(tail.x,tail.y,cx,cy);
    g1.addColorStop(0,   'rgba(255,55,0,0)');
    g1.addColorStop(0.04,'rgba(255,80,0,0.08)');
    g1.addColorStop(0.12,'rgba(255,80,0,0.88)');
    g1.addColorStop(0.42,'rgba(255,40,0,1)');
    g1.addColorStop(0.78,'rgba(255,175,0,1)');
    g1.addColorStop(1,   'rgba(255,235,30,1)');
    c.fillStyle=g1;c.shadowBlur=16;c.shadowColor='rgba(255,110,0,0.85)';
    c.fill();c.restore();

    // Hot white-yellow core — same wave but narrow
    c.save();
    c.beginPath();c.moveTo(tail.x,tail.y);
    for(var i=1;i<h.length;i++){
      var tt=i/(h.length-1),w=W*0.26*tt*tt;
      var wv=Math.sin(tick*0.27+i*0.58)*1.2*tt;
      c.lineTo(h[i].x+px*(w+wv),h[i].y+py*(w+wv));
    }
    for(var i=h.length-2;i>=1;i--){
      var tt=i/(h.length-1),w=W*0.26*tt*tt;
      var wv=Math.sin(tick*0.27+i*0.58+3.14)*1.2*tt;
      c.lineTo(h[i].x-px*(w+wv),h[i].y-py*(w+wv));
    }
    c.closePath();
    var g2=c.createLinearGradient(tail.x,tail.y,cx,cy);
    g2.addColorStop(0,   'rgba(255,255,220,0)');
    g2.addColorStop(0.06,'rgba(255,250,180,0.06)');
    g2.addColorStop(0.18,'rgba(255,250,180,0.82)');
    g2.addColorStop(1,   'rgba(255,255,245,1)');
    c.fillStyle=g2;c.shadowBlur=10;c.shadowColor='rgba(255,255,200,1)';
    c.fill();c.restore();
  }

  function drawRock(c,x,y){
    var R=16;
    var aura=c.createRadialGradient(x,y,R,x,y,R*2.8);
    aura.addColorStop(0,'rgba(255,145,0,0.38)');aura.addColorStop(1,'rgba(255,60,0,0)');
    c.beginPath();c.arc(x,y,R*2.8,0,6.2832);c.fillStyle=aura;c.fill();
    var body=c.createRadialGradient(x-5,y-5,0,x,y,R);
    body.addColorStop(0,'rgba(205,215,245,1)');body.addColorStop(0.32,'rgba(105,115,150,1)');body.addColorStop(1,'rgba(30,36,60,1)');
    c.beginPath();c.arc(x,y,R,0,6.2832);c.fillStyle=body;
    c.shadowBlur=10;c.shadowColor='rgba(255,145,0,0.65)';c.fill();c.shadowBlur=0;
    c.fillStyle='rgba(16,20,44,0.9)';
    c.beginPath();c.arc(x-4.8,y-2.8,3.6,0,6.2832);c.fill();
    c.beginPath();c.arc(x+5.2,y+4.8,2.3,0,6.2832);c.fill();
    c.beginPath();c.arc(x+1.6,y-7,1.6,0,6.2832);c.fill();
    var hl=c.createRadialGradient(x-6,y-6,0,x-6,y-6,9);
    hl.addColorStop(0,'rgba(255,255,255,0.7)');hl.addColorStop(0.5,'rgba(255,255,255,0.2)');hl.addColorStop(1,'rgba(255,255,255,0)');
    c.beginPath();c.arc(x,y,R,0,6.2832);c.fillStyle=hl;c.fill();
    c.strokeStyle='rgba(12,12,35,0.85)';c.lineWidth=2.8;
    c.beginPath();c.arc(x,y,R,0,6.2832);c.stroke();
    c.strokeStyle='rgba(255,165,30,0.95)';c.lineWidth=2;
    c.beginPath();c.arc(x,y,R,0,6.2832);c.stroke();
  }

  function loop(){
    t++;
    mx+=(tx-mx)*0.15;my+=(ty-my)*0.15;
    hist.push({x:mx,y:my});if(hist.length>HIST)hist.shift();
    sparks.push(new Spark(mx,my,true));
    sparks.push(new Spark(mx,my,true));
    if(hist.length>2){
      var dx=hist[hist.length-1].x-hist[hist.length-2].x;
      var dy=hist[hist.length-1].y-hist[hist.length-2].y;
      if(dx*dx+dy*dy>6){sparks.push(new Spark(mx,my,false));sparks.push(new Spark(mx,my,false));}
    }
    if(sparks.length>250)sparks.splice(0,sparks.length-250);
    ctx.clearRect(0,0,cvs.width,cvs.height);
    sparks=sparks.filter(function(s){return s.life>0&&s.sz>0.05;});
    for(var i=0;i<sparks.length;i++){sparks[i].tick();sparks[i].draw(ctx);}
    drawFlame(ctx,hist,mx,my,t);
    // Rock sways on its own axis
    var sx=Math.sin(t*0.14)*3,sy=Math.cos(t*0.11)*2;
    drawRock(ctx,mx+sx,my+sy);
    requestAnimationFrame(loop);
  }
  loop();
})();
</script>""", height=0)

# ── Global CSS: dark space theme + animations ──────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Inter:wght@300;400;600&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background: radial-gradient(ellipse at 20% 50%, #0a0e2e 0%, #020408 60%, #000000 100%);
    color: #e0e8ff;
    font-family: 'Inter', sans-serif;
}
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed; inset: 0; pointer-events: none;
    background-image:
        radial-gradient(1px 1px at 20% 30%, rgba(255,255,255,0.6) 0%, transparent 100%),
        radial-gradient(1px 1px at 80% 10%, rgba(255,255,255,0.4) 0%, transparent 100%),
        radial-gradient(1px 1px at 50% 70%, rgba(255,255,255,0.5) 0%, transparent 100%),
        radial-gradient(1px 1px at 10% 90%, rgba(255,255,255,0.3) 0%, transparent 100%),
        radial-gradient(1px 1px at 90% 60%, rgba(255,255,255,0.4) 0%, transparent 100%);
    z-index: 0;
}
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1340 0%, #060918 100%) !important;
    border-right: 1px solid rgba(100,149,237,0.25);
}
[data-testid="stSidebar"] * { color: #c8d8ff !important; }

.hero-title {
    font-family: 'Orbitron', sans-serif;
    font-size: 2.6rem; font-weight: 900; text-align: center;
    background: linear-gradient(90deg, #4fc3f7, #a78bfa, #f472b6, #4fc3f7);
    background-size: 300% 100%;
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    animation: shimmer 4s linear infinite; margin-bottom: 0.2rem;
}
.hero-sub { text-align: center; color: #7090c0; font-size: 0.85rem; letter-spacing: 0.1em; margin-bottom: 1.5rem; }
@keyframes shimmer { 0%{background-position:0% 50%} 100%{background-position:300% 50%} }

.metric-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.05) 0%, rgba(100,149,237,0.08) 100%);
    border: 1px solid rgba(100,149,237,0.3); border-radius: 16px;
    padding: 1.2rem 1.4rem; text-align: center;
    transition: transform 0.25s ease, box-shadow 0.25s ease;
    animation: fadeUp 0.6s ease both;
}
.metric-card:hover { transform: translateY(-4px); box-shadow: 0 0 24px rgba(100,149,237,0.4); }
.metric-value { font-family: 'Orbitron', sans-serif; font-size: 1.7rem; font-weight: 700; color: #7dd3fc; }
.metric-label { font-size: 0.75rem; color: #8899bb; letter-spacing: 0.08em; margin-top: 0.3rem; }
@keyframes fadeUp { from{opacity:0;transform:translateY(20px)} to{opacity:1;transform:translateY(0)} }

.risk-high {
    background: linear-gradient(135deg, #7f1d1d, #dc2626); border: 1px solid #ef4444;
    border-radius: 12px; padding: 1rem; text-align: center;
    animation: pulse-red 1.5s ease-in-out infinite;
    font-family: 'Orbitron', sans-serif; font-size: 1.1rem; font-weight: 700; color: #fff;
}
.risk-mid {
    background: linear-gradient(135deg, #78350f, #d97706); border: 1px solid #f59e0b;
    border-radius: 12px; padding: 1rem; text-align: center;
    font-family: 'Orbitron', sans-serif; font-size: 1.1rem; font-weight: 700; color: #fff;
}
.risk-low {
    background: linear-gradient(135deg, #052e16, #16a34a); border: 1px solid #22c55e;
    border-radius: 12px; padding: 1rem; text-align: center;
    font-family: 'Orbitron', sans-serif; font-size: 1.1rem; font-weight: 700; color: #fff;
}
@keyframes pulse-red {
    0%,100%{box-shadow:0 0 12px #ef4444} 50%{box-shadow:0 0 32px #ef4444, 0 0 60px rgba(239,68,68,0.53)}
}

[data-testid="stTabs"] button {
    font-family: 'Orbitron', sans-serif !important; font-size: 0.8rem !important;
    color: #6080c0 !important; border-radius: 8px 8px 0 0 !important; transition: all 0.2s !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #7dd3fc !important; border-bottom: 2px solid #7dd3fc !important;
}
h2, h3 { font-family: 'Orbitron', sans-serif !important; color: #a5c8f8 !important; }
.stDataFrame { border-radius: 12px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

PLOTLY_THEME = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(10,14,46,0.6)",
    font=dict(family="Inter", color="#c8d8ff"),
)

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "..", "Possible Asteroid Impacts with Earth NASA Sentry")
IMPACTS_CSV = os.path.join(DATA_DIR, "impacts.csv")
ORBITS_CSV  = os.path.join(DATA_DIR, "orbits.csv")

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

with st.spinner("🛰️ Veriler yükleniyor ve model eğitiliyor…"):
    df_all, df_orbits = load_data()
    model, cv_scores, importances = train_model(df_all)

st.markdown('<div class="hero-title">☄️ Asteroit Takip Gösterge Paneli</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">NASA SENTRY · Possible Asteroid Impacts with Earth</div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🔭 Filtreler")
    year_min = int(df_all["Period Start"].min())
    year_max = int(df_all["Period End"].max())
    year_range = st.slider("📅 Yıl Aralığı", year_min, year_max, (year_min, year_max), step=1)
    diam_min = float(df_all["Asteroid Diameter (km)"].min())
    diam_max = float(df_all["Asteroid Diameter (km)"].max())
    diam_range = st.slider(
        "📏 Asteroid Çapı (km)",
        round(diam_min, 3), round(diam_max, 3),
        (round(diam_min, 3), round(diam_max, 3)),
        step=0.001, format="%.3f",
    )
    sort_by = st.selectbox(
        "⚠️ Tehlikeli Cisimleri Sırala",
        ["Cumulative Impact Probability", "Maximum Torino Scale"],
    )
    st.divider()
    st.caption("🌐 Dashboard · NASA Sentry")

mask = (
    (df_all["Period Start"] >= year_range[0])
    & (df_all["Period End"]   <= year_range[1])
    & (df_all["Asteroid Diameter (km)"] >= diam_range[0])
    & (df_all["Asteroid Diameter (km)"] <= diam_range[1])
)
df = df_all[mask].copy()

tab1, tab2, tab3 = st.tabs([
    "🚨 Tehlike Analizi",
    "🌍 Yörünge Analizi",
    "🤖 Tahmin Modeli",
])

# ── TAB 1 ─────────────────────────────────────────────────────────────────────
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    for col, label, val in [
        (c1, "Filtrelenen Asteroid",  str(len(df))),
        (c2, "Maks. Torino Ölçeği",   str(int(df["Maximum Torino Scale"].max())) if not df.empty else "—"),
        (c3, "Çarpışma Olas. (maks)", f"{df['Cumulative Impact Probability'].max():.2e}" if not df.empty else "—"),
        (c4, "Ort. Hız (km/s)",       f"{df['Asteroid Velocity'].mean():.2f}" if not df.empty else "—"),
    ]:
        col.markdown(
            f'<div class="metric-card"><div class="metric-value">{val}</div>'
            f'<div class="metric-label">{label}</div></div>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.subheader("🚨 Tehlikeli Asteroitler")
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
        st.subheader("🔵 Hız – Boyut Dağılımı")
        if df.empty:
            st.info("Filtreye uyan veri yok.")
        else:
            df["_sz"] = df["Maximum Torino Scale"] + 0.1
            fig_sc = px.scatter(
                df, x="Asteroid Velocity", y="Asteroid Diameter (km)",
                color="Maximum Torino Scale", size="_sz", size_max=28,
                hover_name="Object Name",
                hover_data={"Cumulative Impact Probability": ":.2e",
                            "Period Start": True, "Period End": True, "_sz": False},
                color_continuous_scale="YlOrRd",
                labels={"Asteroid Velocity": "Hız (km/s)",
                        "Asteroid Diameter (km)": "Çap (km)",
                        "Maximum Torino Scale": "Torino"},
                title="Asteroid Hızı ve Boyutu",
            )
            fig_sc.update_layout(**PLOTLY_THEME)
            fig_sc.update_traces(marker=dict(line=dict(width=0.5, color="rgba(255,255,255,0.13)")))
            st.plotly_chart(fig_sc, use_container_width=True)

    with rc:
        st.subheader("📊 En Tehlikeli 10 Asteroid")
        if df.empty:
            st.info("Filtreye uyan veri yok.")
        else:
            top10 = df.sort_values("Cumulative Impact Probability", ascending=False).head(10).copy()
            top10["Label"] = top10["Object Name"].str[:16]
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                name="Olası Çarpışma Sayısı", x=top10["Label"],
                y=top10["Possible Impacts"],
                marker=dict(color=top10["Possible Impacts"],
                            colorscale="Blues", showscale=False,
                            line=dict(width=0)), yaxis="y1",
            ))
            fig_bar.add_trace(go.Scatter(
                name="Çarpışma Olasılığı", x=top10["Label"],
                y=top10["Cumulative Impact Probability"],
                mode="lines+markers",
                marker=dict(size=9, color="#f87171", line=dict(width=1.5, color="#fff")),
                line=dict(color="#f87171", width=2.5), yaxis="y2",
            ))
            fig_bar.update_layout(
                **PLOTLY_THEME,
                title="Olası Çarpışma Sayısı ve Kümülatif Olasılık (Top 10)",
                yaxis=dict(title="Olası Çarpışma Sayısı", side="left", gridcolor="#1e2a4a"),
                yaxis2=dict(title="Çarpışma Olasılığı", overlaying="y", side="right", showgrid=False),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                xaxis_tickangle=-35, bargap=0.25,
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    with st.expander("📋 Tüm Birleşik Veriyi Göster"):
        st.dataframe(df, use_container_width=True)

# ── TAB 2 ─────────────────────────────────────────────────────────────────────
with tab2:
    st.info(
        "**Not:** `impacts.csv` geçici adlar (örn. '2006 WP1'), `orbits.csv` ise "
        "numaralı asteroitler (örn. '433 Eros') içerdiği için iki veri seti "
        "`Object Name` üzerinden eşleşmiyor. Yörünge analizi `orbits.csv` "
        "üzerinden bağımsız olarak yapılmaktadır."
    )
    oa1, oa2 = st.columns(2)
    with oa1:
        st.subheader("🪐 Sınıflandırma Dağılımı")
        cls_df = df_orbits["Object Classification"].value_counts().reset_index()
        cls_df.columns = ["Sinif", "Sayi"]
        fig_pie = px.pie(cls_df, names="Sinif", values="Sayi",
                         title="Asteroid Türleri", hole=0.4,
                         color_discrete_sequence=px.colors.sequential.Plasma_r)
        fig_pie.update_layout(**PLOTLY_THEME)
        fig_pie.update_traces(textfont_size=12, pull=[0.03]*len(cls_df))
        st.plotly_chart(fig_pie, use_container_width=True)

    with oa2:
        st.subheader("📐 Eksantriklik – Eğim")
        orb_c = df_orbits.dropna(subset=["Orbit Eccentricity", "Orbit Inclination (deg)"])
        fig_ec = px.scatter(
            orb_c.sample(min(2000, len(orb_c)), random_state=1),
            x="Orbit Eccentricity", y="Orbit Inclination (deg)",
            color="Object Classification", opacity=0.65,
            labels={"Orbit Eccentricity": "Eksantriklik",
                    "Orbit Inclination (deg)": "Eğim (deg)",
                    "Object Classification": "Tür"},
            title="Yörünge Eksantrikligi ve Eğimi",
            color_discrete_sequence=px.colors.qualitative.Vivid,
        )
        fig_ec.update_layout(**PLOTLY_THEME)
        st.plotly_chart(fig_ec, use_container_width=True)

    st.subheader("📏 Minimum Yörünge Kesişim Mesafesi (MOID)")
    moid = df_orbits["Minimum Orbit Intersection Distance (AU)"].dropna()
    fig_moid = px.histogram(moid, nbins=80, log_y=True,
                             labels={"value": "MOID (AU)", "count": "Sayı (log)"},
                             title="MOID Dağılımı — Dünyaya Yakın NEO'lar",
                             color_discrete_sequence=["#60a5fa"])
    fig_moid.add_vline(x=0.05, line_dash="dash", line_color="#f87171",
                       annotation_text="PHA Eşiği 0.05 AU",
                       annotation_font_color="#f87171")
    fig_moid.update_layout(**PLOTLY_THEME)
    st.plotly_chart(fig_moid, use_container_width=True)

# ── TAB 3 ─────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("🤖 Çarpışma Olasılığı Tahmin Modeli")
    st.markdown(
        f"**Model:** Random Forest Regressor — log₁₀(Cumulative Impact Probability) tahmini  \n"
        f"**Özellikler:** {', '.join(ML_FEATURES)}  \n"
        f"**5-katlı Çapraz Doğrulama R²:** {cv_scores.mean():.3f} ± {cv_scores.std():.3f}"
    )

    mc1, mc2 = st.columns([1, 1])

    with mc1:
        st.subheader("📈 Özellik Önem Puanları")
        imp_df = pd.DataFrame({"Özellik": ML_FEATURES, "Önem": importances})
        imp_df = imp_df.sort_values("Önem", ascending=True)
        fig_imp = px.bar(
            imp_df, x="Önem", y="Özellik", orientation="h",
            color="Önem", color_continuous_scale="ice",
            title="Random Forest — Özellik Önem Sıralaması",
        )
        fig_imp.update_layout(**PLOTLY_THEME, showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig_imp, use_container_width=True)

    with mc2:
        st.subheader("🔮 İnteraktif Tahmin")
        st.caption("Parametreleri ayarlayarak tahmini çarpışma olasılığını hesapla:")
        pred_vel  = st.slider("🚀 Asteroid Hızı (km/s)",
                              float(df_all["Asteroid Velocity"].min()),
                              float(df_all["Asteroid Velocity"].max()),
                              float(df_all["Asteroid Velocity"].median()), step=0.1)
        pred_diam = st.slider("📏 Asteroid Çapı (km)",
                              float(df_all["Asteroid Diameter (km)"].min()),
                              float(df_all["Asteroid Diameter (km)"].max()),
                              float(df_all["Asteroid Diameter (km)"].median()),
                              step=0.001, format="%.3f")
        pred_mag  = st.slider("💡 Parlaklık (Magnitude)",
                              float(df_all["Asteroid Magnitude"].min()),
                              float(df_all["Asteroid Magnitude"].max()),
                              float(df_all["Asteroid Magnitude"].median()), step=0.1)
        pred_pi   = st.slider("💥 Olası Çarpışma Sayısı",
                              int(df_all["Possible Impacts"].min()),
                              int(df_all["Possible Impacts"].max()),
                              int(df_all["Possible Impacts"].median()), step=1)
        pred_pl   = st.slider("📆 Gözlem Periyodu (yıl)",
                              int(df_all["Period Length"].min()),
                              int(df_all["Period Length"].max()),
                              int(df_all["Period Length"].median()), step=1)

        X_pred    = np.array([[pred_vel, pred_diam, pred_mag, pred_pi, pred_pl]])
        log_prob  = model.predict(X_pred)[0]
        pred_prob = 10 ** log_prob

        st.divider()
        rc1, rc2 = st.columns(2)
        rc1.markdown(
            f'<div class="metric-card"><div class="metric-value">{pred_prob:.2e}</div>'
            f'<div class="metric-label">Tahmini Çarpışma Olasılığı</div></div>',
            unsafe_allow_html=True,
        )
        rc2.markdown(
            f'<div class="metric-card"><div class="metric-value">{log_prob:.3f}</div>'
            f'<div class="metric-label">log₁₀(Olasılık)</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)
        if pred_prob >= 1e-3:
            st.markdown('<div class="risk-high">🔴 YÜKSEK RİSK</div>', unsafe_allow_html=True)
        elif pred_prob >= 1e-6:
            st.markdown('<div class="risk-mid">🟡 ORTA RİSK</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="risk-low">🟢 DÜŞÜK RİSK</div>', unsafe_allow_html=True)

    st.divider()
    st.subheader("📊 Gerçek vs Tahmin Edilen Değerler")
    tdf = df_all[ML_FEATURES + ["Cumulative Impact Probability", "Object Name"]].dropna()
    y_act = np.log10(tdf["Cumulative Impact Probability"].values)
    y_hat = model.predict(tdf[ML_FEATURES].values)
    avp_df = pd.DataFrame({
        "Gerçek log10(CIP)": y_act,
        "Tahmin log10(CIP)": y_hat,
        "Object Name": tdf["Object Name"].values,
    })
    fig_avp = px.scatter(
        avp_df, x="Gerçek log10(CIP)", y="Tahmin log10(CIP)",
        hover_name="Object Name", opacity=0.65,
        title="Model Doğruluğu: Gerçek vs Tahmin",
        color_discrete_sequence=["#818cf8"],
    )
    lo = avp_df["Gerçek log10(CIP)"].min()
    hi = avp_df["Gerçek log10(CIP)"].max()
    fig_avp.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi], mode="lines",
        name="Mükemmel Tahmin", line=dict(dash="dash", color="#f87171", width=2),
    ))
    fig_avp.update_layout(**PLOTLY_THEME)
    st.plotly_chart(fig_avp, use_container_width=True)


if __name__ == "__main__":
    import sys, subprocess
    # Guard: only launch when run directly by Python, not inside Streamlit
    if "streamlit.runtime.scriptrunner" not in sys.modules:
        subprocess.run([sys.executable, "-m", "streamlit", "run", __file__])
