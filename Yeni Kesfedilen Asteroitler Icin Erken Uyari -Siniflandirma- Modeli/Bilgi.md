# Yeni Keşfedilen Asteroitler İçin "Erken Uyarı" Sınıflandırma Modeli

> **NASA Sentry** veritabanı üzerinde makine öğrenmesi ile hızlı tehlike tespiti.

---

## Proje Nedir?

NASA'nın **Sentry** sistemi bir asteroitin Dünya ile çarpışma olasılığını hesaplamak için karmaşık yörünge entegrasyonları ve Monte-Carlo simülasyonları çalıştırır. Bu hesaplamalar saatler sürebilir.

Bu proje; yörünge parametrelerini girdi olarak alıp **saniyeler içinde** "Bu asteroit Potansiyel Tehlikeli Asteroit (PHA) sınıfındadır" ya da "Tehlikesizdir" kararını verebilen bir **ikili sınıflandırma modeli** geliştirmeyi amaçlar. Model, fiziksel hesaplamalara gerek kalmadan çalışan bir **ön eleme / erken uyarı katmanı** görevi görür.

---

## Projenin Amacı

| Hedef | Açıklama |
|---|---|
| **Hız** | Sentry'nin saatlik hesabı yerine milisaniyede tahmin |
| **Erken Uyarı** | Yeni keşfedilen bir cismin ilk yörünge verisiyle hemen sınıflandırılması |
| **Ön Eleme** | Tehlikesiz cisimleri hızlıca eleyerek uzman kaynakları gerçek tehlikelere yönlendirme |
| **Şeffaflık** | Hangi parametrenin tahminde en çok rol oynadığını Feature Importance ile görselleştirme |

---

## Nasıl Çalışır?

### Hedef Değişken (Is_Hazardous)
NASA'nın resmi **PHA (Potentially Hazardous Asteroid)** tanımı kullanılır:

```
MOID (Minimum Orbit Intersection Distance) ≤ 0.05 AU  →  Tehlikeli (1)
Aksi halde                                             →  Tehlikesiz (0)
```

### Kullanılan Özellikler (Features)

| Özellik | Açıklama |
|---|---|
| `Orbit Axis (AU)` | Yörünge yarı-büyük ekseni |
| `Orbit Eccentricity` | Yörünge dışmerkezliği (0=daire, 1=parabol) |
| `Orbit Inclination (deg)` | Yörünge eğim açısı |
| `Perihelion Distance (AU)` | Güneş'e en yakın mesafe |
| `Aphelion Distance (AU)` | Güneş'ten en uzak mesafe |
| `Asteroid Magnitude` | Mutlak parlaklık (boyutla ters orantılı) |
| `Classification_Code` | Orbital sınıf: Amor / Apollo / Aten / Apohele |
| `Tisserand_approx` | Jüpiter'e göre Tisserand parametresi (dinamik köken göstergesi) |

> **Data Leakage Koruması:** MOID, Torino Ölçeği, Palermo Ölçeği ve çarpışma olasılığı gibi hedef değişkeni doğrudan sızdıracak sütunlar eğitim verisinden (X) kesinlikle çıkarılmıştır.

### Pipeline Adımları

```
orbits.csv
    │
    ├─► Hedef değişken oluşturma (MOID ≤ 0.05 AU)
    ├─► Özellik mühendisliği (LabelEncoder + Tisserand parametresi)
    ├─► SimpleImputer (medyan stratejisi)
    ├─► Train/Test split (%80 / %20, stratified)
    ├─► StandardScaler
    │
    ├─► Random Forest Classifier (n=300, class_weight=balanced)
    └─► XGBoost Classifier (n=300, scale_pos_weight=1.233)
```

---

## Model Sonuçları

| Model | Accuracy | Precision | **Recall** | F1-Score |
|---|---|---|---|---|
| Random Forest | 0.9341 | 0.9090 | **0.9479** | 0.9280 |
| **XGBoost** | **0.9376** | 0.9052 | **0.9615** | **0.9325** |

> **Recall neden kritik?** Tehlikeli bir asteroiti yanlışlıkla "güvenli" etiketlemek (False Negative), güvenli bir asteroiti "tehlikeli" olarak etiketlemekten çok daha büyük bir risk taşır. Bu nedenle **Recall metriği** model seçiminde önceliklidir.

**XGBoost kazanıyor:** 1401 tehlikeli asteroitin yalnızca ~54'ünü kaçırıyor (Recall = %96.15).

---

## Çıktı Dosyaları

| Dosya | İçerik |
|---|---|
| `erken_uyari_modeli.py` | Tüm pipeline'ı çalıştıran ana script |
| `feature_importance.png` | Her iki modelin özellik önem sıralaması |
| `confusion_matrix.png` | Test seti karışıklık matrisleri (FN kırmızıyla vurgulanmış) |
| `metric_comparison.png` | Accuracy / Precision / Recall / F1 yan yana karşılaştırma |

---

## Kullanılan Teknolojiler

| Teknoloji | Versiyon | Kullanım Amacı |
|---|---|---|
| **Python** | 3.13.5 | Ana programlama dili |
| **Pandas** | 2.3.3 | Veri yükleme, birleştirme, işleme |
| **NumPy** | 2.1.3 | Sayısal hesaplamalar, Tisserand parametresi |
| **Scikit-Learn** | 1.6.1 | Imputer, Scaler, Train/Test split, Random Forest, metrikler |
| **XGBoost** | 3.4.0 | Gradient boosting sınıflandırıcı |
| **Matplotlib** | 3.10.0 | Grafik çizimi |
| **Seaborn** | 0.13.2 | Karışıklık matrisi ısı haritası |

---

## Veri Seti

**Kaynak:** [NASA Sentry — Possible Asteroid Impacts with Earth](https://cneos.jpl.nasa.gov/sentry/)

| Dosya | Satır | Açıklama |
|---|---|---|
| `orbits.csv` | 15.635 | Tüm Yakın-Dünya Asteroitlerinin (NEO) yörünge parametreleri |
| `impacts.csv` | 683 | Potansiyel çarpışma yörüngesi hesaplanan cisimler (ayrı format) |

> **Not:** `impacts.csv` ile `orbits.csv` içindeki `Object Name` formatları birbiriyle eşleşmediğinden (biri provisional designation, diğeri numaralı/isimli asteroit) model yalnızca `orbits.csv` üzerinde eğitilmiştir.

---

## Nasıl Kullanılır?

### Gereksinimler

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn
```

### Çalıştırma

```bash
python erken_uyari_modeli.py
```

Script çalıştırıldığında sırasıyla şunları yapar:

1. `orbits.csv` dosyasını yükler
2. MOID kriterine göre `Is_Hazardous` etiketini oluşturur
3. Özellik mühendisliği uygular
4. Veriyi temizler ve ölçeklendirir
5. Random Forest ve XGBoost modellerini eğitir
6. Konsola metrik raporunu yazar
7. Üç grafik dosyası üretir ve klasöre kaydeder

---

## Nerede Kullanılır?

- **Asteroит izleme merkezleri** — Yeni keşfedilen bir cisim için ilk hızlı değerlendirme
- **Teleskop gözlem önceliklendirmesi** — Sınırlı teleskop zamanını gerçek tehditlere tahsis etme
- **Eğitim & simülasyon** — Gezegen savunması senaryolarında karar destek sistemi
- **Araştırma** — MOID veya fiziksel parametreler ile tehlike korelasyonlarını keşfetme
- **API entegrasyonu** — Flask/FastAPI ile sarılarak yörünge verisi POST edilen bir tahmin servisi olarak kullanılabilir