# NASA Asteroid Risk ve Tehlike Analizi Projeleri

Bu klasör, NASA açık verilerini kullanarak asteroid tehditlerini analiz eden, tahmin eden ve görselleştiren dört farklı proje içerir. Tüm projeler aynı temel veri kaynağına dayanır: NASA Sentry verileri, asteroid yörünge bilgileri ve çarpışma risk parametreleri. Amaçları farklı olsa da hepsi aynı büyük hedefe hizmet eder: uzayda potansiyel tehlike oluşturabilecek cisimleri daha erken tanımak, daha iyi öngörmek ve sonuçları anlaşılır şekilde sunmak.

---

## 1) Dashboard Etkileşimli Asteroit Takip Gösterge Paneli

### Kısa açıklama
Bu proje, NASA Sentry verilerini canlı ve etkileşimli bir arayüz üzerinden kullanıcıya sunan bir izleme panosudur. Kullanıcılar basinçlı bir veri setinin içinden önemli risk göstergelerini filtreleyebilir, asteroidleri farklı kriterlere göre değerlendirebilir ve yörünge davranışlarını görselleştirebilir.

### Projede neler yapılır?
- Asteroidleri çarpışma riskine göre inceleme
- Hız, çap, Torino ölçeği ve benzeri bilgiler üzerinden ön değerlendirme
- Yörünge dağılımı, MOID ve sınıflandırma analizleri
- Makine öğrenmesiyle çarpışma olasılığı ve etkisi tahmini
- Kullanıcı dostu, uzay temalı interactive dashboard

### Kullanılan teknolojiler
- Python
- Streamlit
- Pandas
- NumPy
- Plotly
- scikit-learn
- HTML5 Canvas
- JavaScript
- CSS ve özel görsel efektler

### Yenilik / değer
Bu proje, sadece veri analizi değil; karar vericiler, araştırmacılar ve eğitim amaçlı kullanıcılar için doğrudan kullanışlı bir araçtır. Uzay verisini popüler ve anlaşılır bir arayüze taşıması nedeniyle güçlü bir "gösterge paneli" yaklaşımı sunar. Görsel ve etkileşimli yapısı, asteroit risklerini daha kolay fark etmeyi sağlar.

---

## 2) Yeni Keşfedilen Asteroitler İçin Erken Uyarı - Sınıflandırma Modeli

### Kısa açıklama
Bu proje, yeni keşfedilen asteroitlerin erken aşamada "tehlikeli" ya da "güvenli" olarak sınıflandırılmasını sağlayan bir makine öğrenmesi modelidir. Temel fikir, karmaşık fiziksel yörünge hesaplamalarına girmeden, ilk parametrelerle hızlı bir değerlendirme yapmaktır.

### Projede neler yapılır?
- Yörünge verilerini kullanarak tehlike etiketi üretme
- Potansiyel Tehlikeli Asteroit (PHA) tanımına göre sınıflandırma
- Random Forest ve XGBoost ile karşılaştırmalı model eğitimi
- Reklam gibi değil, gerçek risk değerlendirme mantığına dayalı sınıflandırma
- Recall odaklı değerlendirme ile yanlış negatif risklerini azaltma

### Kullanılan teknolojiler
- Python
- Pandas
- NumPy
- scikit-learn
- XGBoost
- Matplotlib
- Seaborn

### Yenilik / değer
Bu proje en kritik yönüyle "erken uyarı sistemi" fikrini temsil eder. Yeni bir asteroid keşfedildiğinde, kapsamlı simülasyonların yapılmasına gerek kalmadan hızlı bir karar destek mekanizması kurar. Özellikle tehlikeli bir asteroitin güvenli olarak etiketlenmesi gibi hatalarla ilgili olarak Recall önceliği, bu modelin pratik ve güvenli kullanımına büyük katkı sağlar.

---

## 3) Yıkıcılık ve Etki Tahminleyicisi - Regresyon Modeli

### Kısa açıklama
Bu proje, asteroidin çapı ve hızı gibi temel özelliklerine dayanarak çarpışma sırasında oluşacak enerji düzeyini ve Torino ölçeğini tahmin etmeye çalışır. Yani asteroitin yalnızca "tehlikeli mi değil mi" sorusundan öte, "ne kadar yıkıcı olabileceği" sorusunu ele alır.

### Projede neler yapılır?
- Çaptan ve hızdan kütle ve kinetik enerji hesaplama
- Joule ve megaton TNT cinsinden enerji tahmini
- Random Forest regresyonu ile yıkıcılık modelleme
- Torino ölçeği gibi risk değeri yaklaşımı
- Fiziksel ilişkiler ile makine öğrenmesi birleşimi

### Kullanılan teknolojiler
- Python
- Jupyter Notebook
- NumPy
- Pandas
- Matplotlib
- scikit-learn

### Yenilik / değer
Bu proje, "tehlike tespiti" ile "etki büyüklüğü tahmini" arasındaki farkı net gösterir. Asteroidin sadece riskli olup olmadığını değil, potansiyel yıkım gücünü de anlamaya çalışır. Bilimsel olarak da oldukça anlamlıdır çünkü çap ve hızın enerjiye dönüşümünü fizik formülleriyle birlikte görselleştirir.

---

## 4) Zaman Serisi Analizi ve Tehlike Takvimi

### Kısa açıklama
Bu proje, asteroid risklerinin zaman içindeki değişimini yıllar boyunca inceleyen bir zaman serisi analizi çalışmasıdır. Gelecekteki 100 yıl içinde hangi dönemlerin daha riskli olabileceğini, hangi yıllarda daha fazla tehlike sinyali oluşabileceğini görselleştirir.

### Projede neler yapılır?
- Yıllık risk serisi oluşturma
- ADF testi ile durağanlık analizi
- ARIMA ile zaman serisi tahmini
- Geleceğe dair risk takvimi üretme
- Plotly ile interaktif grafikler ve Seaborn ile ısı haritası

### Kullanılan teknolojiler
- Python
- Pandas
- NumPy
- statsmodels
- Matplotlib
- Seaborn
- Plotly
- ARIMA / zaman serisi metodolojisi

### Yenilik / değer
Bu proje, tek bir asteroid değil, tüm risk penceresinin zamansal eğilimini analiz eder. Başka bir deyişle projede "tehlike" sadece anlık bir durum değil, zaman boyunca değişen bir eğilim olarak değerlendirilir. Bu yaklaşım, geleceğe ilişkin senaryo üretmek için güçlü bir yöntem sunar.

---

## Genel bakış: bu proje setinin güçlü yönü

Bu dört proje birlikte değerlendirildiğinde, NASA asteroid verilerinin yalnızca bir veri seti değil; çok katmanlı bir risk yönetim ve analitik ekosistemi oluşturduğunu gösterir.

- Birinci proje: görünür ve kullanıcı dostu izleme arayüzü
- İkinci proje: hızlı erken uyarı ve sınıflandırma
- Üçüncü proje: fiziksel etkilenme gücü ve yıkıcılık tahmini
- Dördüncü proje: zaman boyutunda gelecek risk takvimi

Bu kombinasyon, asteroit tehditlerine yaklaşımın sadece bir model sorunu değil; veri görselleştirme, fiziksel yorumlama, makine öğrenmesi ve zaman serisi analizi gibi birbirini tamamlayan bir sistem olduğunu ortaya koyar.

### Temel kazanımlar
- NASA verilerini bilimsel ve pratik şekilde işlemede deneyim kazanmak
- Makine öğrenmesi ile gerçek dünya risk analizleri yapmak
- Görselleştirme ve dashboard tasarımını öğrenmek
- Fizik, veri bilimi ve astrofizik arasındaki bağlantıyı görsel olarak anlamak
- Gelecek odaklı risk yönetimi için örnek bir analitik yöntem geliştirmek

---

## Sonuç

Bu klasör, asteroit tehditlerine yaklaşımın dört farklı ama uyumlu boyutunu bir araya getirir:

- takibi,
- erken uyarıyı,
- etki tahminini,
- gelecek risk zaman çizelgesini.

Bu da projeyi sadece bir "veri analizi çalışması" olmaktan çıkarıp, gerçek dünya uzay güvenliği ve risk değerlendirme bağlamında anlamlı bir portfolio örneği haline getirir.
