# 🚀 Dashboard — Etkileşimli Asteroit Takip Gösterge Paneli

NASA Sentry verilerini kullanarak olası Dünya'ya çarpabilecek asteroitleri görselleştiren, makine öğrenimi ile çarpışma olasılığı tahmin eden interaktif bir web panelidir.

---

## 📌 Ne Yapar?

### 1. Tehlike Analizi Sekmesi
- NASA Sentry veri setindeki tüm asteroidleri listeler
- **4 metrik kart**: Filtrelenen asteroid sayısı, maksimum Torino Ölçeği, maksimum çarpışma olasılığı, ortalama hız
- **Hız–Boyut Scatter Plot**: Asteroid hızı ile çapı arasındaki ilişkiyi Torino Ölçeğine göre renklendirir
- **Top 10 Bar + Line Chart**: En tehlikeli 10 asteroitin olası çarpışma sayısı ve kümülatif olasılığını çift eksenli grafikte gösterir
- Sidebar filtrelerle yıl aralığı, çap aralığı ve sıralama kriteri seçilebilir

### 2. Yörünge Analizi Sekmesi
- `orbits.csv` üzerinden bağımsız yörünge analizi yapar
- **Pasta Grafik**: Asteroid sınıflandırmalarının dağılımı (Apollo, Amor, Aten vb.)
- **Eksantriklik–Eğim Scatter**: Yörünge şekillerinin dağılımı
- **MOID Histogramı**: Minimum Yörünge Kesişim Mesafesi dağılımı, PHA eşiği (0.05 AU) çizgisi ile

### 3. Tahmin Modeli Sekmesi
- **Random Forest Regressor** ile `log₁₀(Kümülatif Çarpışma Olasılığı)` tahmini
- 5-katlı çapraz doğrulama R² skoru ile model performansı raporlanır
- **Özellik önem grafiği**: Hangi parametreler tahmine daha çok katkı sağlıyor
- **İnteraktif sliderlar**: Hız, çap, parlaklık, çarpışma sayısı, gözlem periyodu girilerek anlık tahmin
- Risk seviyesi (Yüksek / Orta / Düşük) animasyonlu badge ile gösterilir
- Gerçek vs Tahmin edilen değerlerin dağılım grafiği

---

## 🛠️ Kullanılan Teknolojiler

| Teknoloji | Kullanım Amacı |
|---|---|
| **Python 3** | Ana programlama dili |
| **Streamlit** | Web arayüzü ve dashboard çerçevesi |
| **Pandas** | CSV veri okuma, filtreleme, birleştirme |
| **NumPy** | Sayısal hesaplamalar, log dönüşümleri |
| **Plotly Express / Graph Objects** | İnteraktif grafikler (scatter, bar, pie, histogram) |
| **scikit-learn** | Random Forest modeli, StandardScaler pipeline, cross_val_score |
| **streamlit.components.v1** | Özel JavaScript/Canvas enjeksiyonu |
| **HTML5 Canvas API** | Fare üzerine animasyonlu meteor cursor |
| **JavaScript (Vanilla)** | Cursor animasyonu: konum geçmişi, sparkle partiküller, alev çizimi |
| **CSS (Google Fonts)** | Orbitron + Inter fontları, shimmer/fadeUp/pulse-red animasyonları |

---

## 📂 Veri Setleri

- `impacts.csv` — NASA Sentry olası çarpışma tahminleri (asteroid adı, hız, çap, çarpışma olasılığı, Torino/Palermo ölçeği vb.)
- `orbits.csv` — Asteroid yörünge parametreleri (eksantriklik, eğim, MOID, sınıflandırma vb.)

> **Not:** İki veri seti `Object Name` üzerinden birleştirilir ancak geçici adlar (örn. `2006 WP1`) ile numaralı adlar (örn. `433 Eros`) birbiriyle eşleşmediğinden yörünge analizi `orbits.csv` üzerinden bağımsız çalışır.

---

## 🎨 Arayüz Özellikleri

- **Koyu uzay teması**: Radial gradient arka plan, yıldız nokta efektleri
- **Animasyonlu başlık**: Shimmer (kayan renk) efektli gradient metin
- **Metrik kartlar**: Hover'da yükselen, fadeUp ile açılan kartlar
- **Risk badge**: Yüksek risk durumunda kırmızı nabız (pulse) animasyonu
- **Meteor cursor**: Fare imleci yerine HTML5 Canvas üzerinde çizilen animasyonlu meteor
  - 42 noktalık konum geçmişiyle oluşturulan alev kuyruğu
  - Alev dalganın iki katmanı: ana kırmızı-turuncu alev + beyaz-sarı sıcak öz
  - Trail boyunca dairesel yumuşak glow blob'ları (keskin kenar olmadan)
  - 4 köşeli karikatür yıldız kıvılcımlar (quadratic eğri, glow efektli)
  - Kaya başı kendi ekseni etrafında sarsılır

---

## ▶️ Çalıştırmak İçin

```bash
python -m streamlit run app.py
```

Tarayıcıda aç: [http://localhost:8501](http://localhost:8501)

---

## 📊 Makine Öğrenimi Detayları

- **Model**: `RandomForestRegressor(n_estimators=200, random_state=42)`
- **Pipeline**: `StandardScaler → RandomForestRegressor`
- **Hedef değişken**: `log₁₀(Cumulative Impact Probability)` (log dönüşümü ile normalleştirme)
- **Özellikler**:
  - Asteroid Velocity
  - Asteroid Diameter (km)
  - Asteroid Magnitude
  - Possible Impacts
  - Period Length
- **Değerlendirme**: 5-katlı çapraz doğrulama R² skoru
