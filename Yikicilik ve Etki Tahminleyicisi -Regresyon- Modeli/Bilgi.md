# Yıkıcılık ve Etki Tahminleyicisi

NASA Sentry verilerini kullanarak olası bir asteroid çarpışmasının fiziksel enerji büyüklüğünü ve Torino ölçeği değerini tahmin eden Jupyter Notebook tabanlı bir makine öğrenmesi projesidir.

Projenin temel sorusu şudur:

> **"Hangi asteroid olası bir çarpışmada daha yüksek yıkıcı etki oluşturabilir?"**

Model, kullanıcının yalnızca asteroid çapını metre ve hızını km/s cinsinden vermesiyle yaklaşık olarak:

- Kinetik enerjiyi megaton TNT eşdeğerinde,
- Torino Çarpma Tehlikesi Ölçeği skorunu

tahmin eder.

Bu çalışma bilimsel ve eğitim amaçlı bir risk/etki tahmin prototipidir. Gerçek bir erken uyarı veya resmi gezegen savunma sistemi değildir.

## Proje ne işe yarar?

Asteroidin fiziksel özellikleri ile olası çarpışma enerjisi arasındaki ilişkiyi görünür hale getirir. Özellikle çapın enerji üzerindeki güçlü etkisini, hızın karesel etkisini ve makine öğrenmesiyle tahmin üretme sürecini göstermek için kullanılabilir.

Proje şu amaçlara hizmet eder:

- NASA Sentry asteroid verilerini analiz etmek,
- Ham veriyi makine öğrenmesine uygun hale getirmek,
- Çap ve hızı fiziksel özelliklere dönüştürmek,
- Kinetik enerjiyi Joule ve megaton TNT cinsinden hesaplamak,
- Doğrusal olmayan ilişkileri Random Forest regresyonu ile modellemek,
- Bir asteroidin göreli yıkıcı etkisini farklı nesneler arasında karşılaştırmak,
- Veri bilimi, fizik ve makine öğrenmesi kavramlarını tek bir örnekte birleştirmek.

## Kullanılan veri setleri

Notebook iki CSV dosyasını okur:

- `impacts.csv`: NASA Sentry olası çarpışma kayıtlarıdır. Asteroid adı, olası çarpışma dönemi, çarpışma sayısı, kümülatif çarpışma olasılığı, hız, çap, Palermo ve Torino ölçeği alanlarını içerir.
- `orbits.csv`: Asteroidlerin yörünge ve fiziksel/astronomik bilgilerini içerir. Yörünge sınıfı, yarı büyük eksen, dışmerkezlik, eğim, günberi/ günöte uzaklığı, yörünge periyodu ve minimum yörünge kesişme mesafesi gibi alanlar bulunur.

Notebook, dosyaları önce çalışma dizininde arar. Dosyalar orada yoksa şu kaynak klasöre yönelir:

```text
../Possible Asteroid Impacts with Earth NASA Sentry
```

Bu nedenle notebook, mevcut klasör yapısında doğrudan açılıp çalıştırılabilir.

## Notebook akışı

### 1. Kütüphanelerin ve verilerin yüklenmesi

İlk Python hücresinde şu teknolojiler içe aktarılır:

- `pathlib.Path`: Dosya yollarını işletim sisteminden bağımsız yönetmek için,
- `re`: Asteroid adlarındaki boşluk karakterlerini standartlaştırmak için,
- `numpy`: Sayısal işlemler ve matematiksel hesaplar için,
- `pandas`: CSV okuma, tablo birleştirme ve veri analizi için,
- `matplotlib`: Dağılım histogramlarını çizmek için,
- `scikit-learn`: Veri bölme, eksik veri doldurma, Random Forest, pipeline ve metrikler için.

`pd.read_csv()` ile `impacts.csv` ve `orbits.csv` okunur; şekilleri ve ilk kayıtları yazdırılarak veri yapısı kontrol edilir.

### 2. Veri birleştirme ve EDA

İki tabloda ortak alan olarak `Object Name` kullanılır. Yörünge dosyasındaki isimlerde normal boşluk yerine `non-breaking space` bulunabileceği için `normalize_name()` fonksiyonu:

1. Özel boşluk karakterlerini normal boşluğa çevirir,
2. Birden fazla boşluğu tek boşluğa indirir,
3. Baş ve sondaki boşlukları siler,
4. Karşılaştırmayı büyük/küçük harfe duyarsız hale getirir.

Ardından `merge_key` üzerinden şu birleştirme yapılır:

```python
merged = impacts.merge(
    orbits.drop_duplicates("merge_key"),
    on="merge_key",
    how="left",
    indicator=True
)
```

`left merge`, bütün Sentry impact kayıtlarının korunmasını sağlar. `indicator=True` ise her kaydın iki tabloda da bulunup bulunmadığını gösterir.

Gerçek veriyle yapılan çalıştırmada:

- `impacts.csv`: 683 kayıt,
- `orbits.csv`: 15.635 kayıt,
- Normalize edilmiş adlarla eşleşme oranı: yaklaşık `%0,44`.

Bu düşük oran, iki CSV’nin aynı kapsam ve isimlendirme döneminden üretilmemiş olabileceğini gösterir. Modelin enerji hesabı impact tablosundaki çap ve hız alanlarıyla çalıştığı için yörünge eşleşmesi az olsa da notebook çalışmaya devam eder.

EDA bölümünde ayrıca:

- Her sütundaki eksik değer sayısı incelenir,
- Sayısal alanlar `pd.to_numeric(..., errors="coerce")` ile güvenli biçimde sayıya çevrilir,
- Çap, hız ve Torino skoru dağılımları histogramlarla görselleştirilir,
- `describe()` ile temel istatistikler hesaplanır.

### 3. Özellik mühendisliği ve fiziksel hesaplama

Veri setinde doğrudan asteroid kütlesi bulunmadığı için kütle çap üzerinden tahmin edilir. Asteroid küre kabul edilir ve sabit yoğunluk varsayılır:

```text
Yoğunluk = 3000 kg/m³
```

Çap kilometreden metreye, hız kilometre/saniyeden metre/saniyeye çevrilir.

Kürenin hacmi:

```text
V = π / 6 × d³
```

Kütle:

```text
m = V × ρ
```

Kinetik enerji:

```text
Eₖ = 1/2 × m × v²
```

Megaton TNT dönüşümü:

```text
1 megaton TNT = 4,184 × 10¹⁵ Joule
Enerji (megaton TNT) = Enerji (Joule) / 4,184 × 10¹⁵
```

Kodda oluşturulan temel özellikler şunlardır:

- `diameter_m`: Metre cinsinden çap,
- `velocity_m_s`: Metre/saniye cinsinden hız,
- `volume_m3`: Tahmini asteroid hacmi,
- `mass_kg`: Tahmini kütle,
- `kinetic_energy_joule`: Joule cinsinden kinetik enerji,
- `energy_megaton_tnt`: Megaton TNT eşdeğeri.

Çap ve hızdaki eksik değerler medyan ile doldurulur. Medyan, aşırı büyük veya küçük asteroid değerlerinden ortalamaya göre daha az etkilenir.

Torino hedefi için `Maximum Torino Scale` sayısala çevrilir, eksik değerler `0` kabul edilir ve skor `0-10` aralığına kırpılır:

```python
merged["torino_score"] = (
    pd.to_numeric(merged["Maximum Torino Scale"], errors="coerce")
    .fillna(0)
    .clip(0, 10)
)
```

### 4. Model girdileri ve hedefler

Kullanıcının tahmin fonksiyonunda yalnızca çap ve hız girebilmesiyle tutarlı olmak için model iki bağımsız değişken kullanır:

```python
FEATURES = ["diameter_m", "velocity_m_s"]
```

İki ayrı bağımlı değişken vardır:

- `energy_megaton_tnt`: Regresyonla tahmin edilen fiziksel enerji,
- `torino_score`: Regresyonla tahmin edilen 0-10 arası Torino skoru.

Veri `%80` eğitim ve `%20` test olacak şekilde `train_test_split()` ile ayrılır. `random_state=42`, aynı kodun tekrar çalıştırıldığında aynı bölmeyi üretmesini sağlar.

Bu sürümde `StandardScaler` kullanılmamıştır. Bunun nedeni Random Forest’ın ağaç tabanlı olması ve özelliklerin ölçeklerinden, örneğin lineer modeller kadar, etkilenmemesidir. `SimpleImputer` yine de model pipeline’ında tutulur; böylece model girdilerindeki eksik değerler eğitim medyanıyla doldurulur.

### 5. Random Forest regresyonu

Model olarak `RandomForestRegressor` kullanılır:

```python
RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
    min_samples_leaf=2
)
```

Random Forest, çok sayıda karar ağacının tahminlerini birleştirir. Bu proje için uygundur çünkü:

- Çap ve hızın enerjiyle ilişkisi doğrusal değildir,
- Kinetik enerji hızın karesiyle değişir,
- Farklı özellik aralıklarıyla çalışabilir,
- Tek bir karar ağacına göre daha kararlı sonuç verebilir.

Enerji ve Torino için ayrı pipeline/model eğitilir:

```python
energy_model = make_model().fit(X_train, y_energy_train)
torino_model = make_model().fit(X_train, y_torino_train)
```

### 6. Model değerlendirme metrikleri

Test verisi üzerinde üç metrik hesaplanır:

- **RMSE (Root Mean Squared Error):** Hataların karelerinin ortalamasının kareköküdür. Büyük hataları daha fazla cezalandırır.
- **MAE (Mean Absolute Error):** Tahmin ve gerçek değer arasındaki mutlak farkın ortalamasıdır. Ortalama hatayı doğrudan yorumlamak için kullanılır.
- **R² (R-squared):** Modelin hedef değişkendeki değişimin ne kadarını açıkladığını gösterir. `1` ideal uyumu, `0` ortalama tahminiyle benzer performansı, negatif değerler ise zayıf uyumu gösterebilir.

Notebook’un gerçek veriyle çalıştırılmış çıktısı:

```text
Megaton TNT: RMSE=2513.7446 | MAE=232.9480 | R²=0.7155
Torino:      RMSE=0.0000    | MAE=0.0000   | R²=1.0000
```

Torino sonuçları dikkatli yorumlanmalıdır. Veri setindeki Torino değerleri büyük ölçüde `0` olduğu için model çoğunlukla sabit bir hedefi öğrenmiştir. Bu sonuç Torino risk tahmininin gerçek hayatta mükemmel olduğu anlamına gelmez.

Enerji tahminindeki RMSE’nin MAE’den çok büyük olması, bazı büyük asteroid örneklerinin hatayı kuvvetle etkilediğini düşündürür. Enerji değerleri çok geniş bir aralıkta değiştiği için gelecekte `log1p(energy_megaton_tnt)` hedefiyle ikinci bir model denenebilir.

## Nasıl kullanılır?

### Gereksinimler

Python 3 ve Jupyter Notebook/JupyterLab gerekir. Kütüphaneler:

```bash
pip install numpy pandas matplotlib scikit-learn jupyter
```

### Çalıştırma adımları

1. `impacts.csv` ve `orbits.csv` dosyalarını kaynak veri klasöründe tutun.
2. `yikicilik_ve_etki_tahminleyicisi.ipynb` dosyasını Jupyter veya VS Code Notebook görünümünde açın.
3. Hücreleri yukarıdan aşağıya sırayla çalıştırın.
4. Model eğitim hücresinin tamamlanmasını bekleyin.
5. Son hücredeki tahmin fonksiyonunu kendi değerlerinizle çağırın.

### Tahmin fonksiyonu

```python
sonuc = yikicilik_tahmin_et(
    cap=100,   # metre
    hiz=20     # km/s
)

print(sonuc)
```

Fonksiyon şu yapıda sonuç döndürür:

```python
{
    "megaton_tnt_tahmini": 70.47,
    "torino_skoru_tahmini": 0.0
}
```

Fonksiyon, çap veya hız `0` ya da negatif olduğunda `ValueError` üretir. Torino tahmini `0-10`, enerji tahmini ise negatif olmayacak şekilde sınırlandırılır.

## Kullanım alanları

Bu prototip aşağıdaki senaryolarda kullanılabilir:

- Eğitim ve ders projelerinde regresyon modelleme örneği olarak,
- NASA açık verileriyle keşifçi veri analizi çalışmaları için,
- Asteroidleri tahmini enerji büyüklüğüne göre sıralamak için,
- Fiziksel formüller ile makine öğrenmesi sonuçlarını karşılaştırmak için,
- Veri ön işleme, eksik veri doldurma ve model değerlendirme pratiği için,
- Bilim iletişimi ve etkileşimli asteroid risk demoları için.

Bir dashboard veya web uygulamasına dönüştürüldüğünde kullanıcıdan çap ve hız alınarak anlık sonuç gösterilebilir. Ancak sonuçlar karar destek veya resmi risk bildirimi olarak kullanılmamalıdır.

## Bilimsel ve teknik sınırlamalar

1. **Yoğunluk sabittir:** Gerçek asteroidlerin yoğunluğu bileşimine, gözenekliliğine ve iç yapısına göre değişir.
2. **Şekil küre kabul edilir:** Düzensiz şekilli cisimlerde küre varsayımı kütle hesabını değiştirebilir.
3. **Atmosfer hesaba katılmaz:** Küçük cisimlerde atmosferik parçalanma ve enerji kaybı önemli olabilir.
4. **Çarpışma açısı ve konumu hesaba katılmaz:** Etki; açı, zemin, okyanus/kara ve konuma göre değişebilir.
5. **Hız alanı basitleştirilmiştir:** Kullanılan hız, veri setindeki asteroid hızıdır; her kayıt için Dünya’ya göre gerçek çarpma hızıyla aynı olmayabilir.
6. **Torino hedefi dengesizdir:** Torino skorlarının çoğu `0` olduğundan mevcut model risk sınıflarını iyi ayıramaz.
7. **Yörünge eşleşmesi düşüktür:** İki dosyanın ortak ad eşleşmesi gerçek çalıştırmada yaklaşık `%0,44` olmuştur. Bu nedenle yörünge alanları model girdisi olarak kullanılmamıştır.
8. **Regresyon, resmi Torino hesaplamasının yerine geçmez:** Torino ölçeği yalnızca enerjiye değil, çarpma olasılığına, cismin boyutuna ve olayın arka plan riskine de bağlıdır.
9. **Model dışı girdiler:** Eğitim verisindeki aralığın çok dışında çap veya hız verilirse Random Forest güvenilir fiziksel ekstrapolasyon yapamaz.

## Geliştirme önerileri

- Güncel NASA Sentry API/verileriyle düzenli veri yenileme,
- Asteroid adları yerine güvenilir benzersiz kimlik kullanımı,
- Palermo skoru ve çarpma olasılığını ek model girdileri olarak değerlendirme,
- Torino skorlarında `0` dışındaki örnekleri artırma veya sınıflandırma yaklaşımı kullanma,
- Eğitim/test ayrımına ek olarak çapraz doğrulama ve belirsizlik aralıkları ekleme,
- Enerji hedefinde logaritmik dönüşüm ve hata dağılımı analizi,
- Yoğunluk için asteroid türüne göre aralık veya olasılıksal model kullanma,
- Sonuçları Streamlit, Flask veya FastAPI tabanlı bir arayüzle sunma.

## Dosya yapısı

```text
Yikicilik ve Etki Tahminleyicisi -Regresyon- Modeli/
├── Bilgi.md
└── yikicilik_ve_etki_tahminleyicisi.ipynb
```

Veri dosyaları notebook’un beklediği kaynak klasördedir:

```text
Nasa/
├── Possible Asteroid Impacts with Earth NASA Sentry/
│   ├── impacts.csv
│   └── orbits.csv
└── Yikicilik ve Etki Tahminleyicisi -Regresyon- Modeli/
    ├── Bilgi.md
    └── yikicilik_ve_etki_tahminleyicisi.ipynb
```

## Teknoloji özeti

| Teknoloji | Kullanıldığı yer | Görevi |
|---|---|---|
| Python | Notebook’un tamamı | Veri işleme, fizik hesapları ve modelleme |
| Jupyter Notebook | `.ipynb` dosyası | Kodu hücre hücre çalıştırma ve sonuçları saklama |
| pandas | Veri yükleme, merge ve EDA | Tablo verilerini yönetme |
| NumPy | Hacim, kütle ve enerji hesapları | Sayısal/matematiksel işlemler |
| Matplotlib | EDA histogramları | Veri dağılımlarını görselleştirme |
| scikit-learn | Pipeline, veri bölme, Random Forest ve metrikler | Makine öğrenmesi süreci |
| RandomForestRegressor | Enerji ve Torino modelleri | Doğrusal olmayan regresyon |
| SimpleImputer | Model pipeline’ı | Eksik girişleri medyanla doldurma |

## Sonuç

Bu proje, NASA Sentry verilerinden başlayan ve fiziksel hesaplamalarla desteklenen uçtan uca bir regresyon prototipidir. En güçlü ve doğrudan çıktısı, çap ve hızdan tahmin edilen kinetik enerjinin megaton TNT cinsinden ifade edilmesidir. Torino tahmini ise veri dağılımındaki `0` ağırlığı nedeniyle şu an deneysel niteliktedir ve daha dengeli, daha güncel risk verileriyle geliştirilmelidir.
