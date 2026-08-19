# Zaman Serisi Analizi ve Tehlike Takvimi (2026-2126)

NASA Sentry "Possible Asteroid Impacts with Earth" veri setini (`impacts.csv`, `orbits.csv`) kullanarak gelecek 100 yil icin bir asteroid tehlike takvimi olusturan, uctan uca bir zaman serisi (time series) analizi ve tahminleme projesi.

## Projenin Amaci

Dunya icin potansiyel tehdit olusturan asteroidlerin bilinen yakin gecis/carpisma risk pencerelerini analiz ederek, onumuzdeki 100 yil (2026-2126) icinde:

- Hangi yillarin/donemlerin daha yuksek riskli oldugunu sayisal bir "Tehlike Skoru" ile ortaya koymak,
- Kesif hizinin (yeni risk listesine giren asteroid sayisinin) zaman icindeki trendini modelleyip gelecege projekte etmek,
- Sonuclari etkilesimli grafikler ve isi haritasi ile gorsellestirerek "gokyuzunun en hareketli olacagi" donemleri gostermek

amaclanmistir.

## Kullanilan Teknolojiler ve Kullanim Amaclari

| Teknoloji | Nerede / Nasil Kullanildi | Amac |
|---|---|---|
| **Python 3** | Tum proje | Veri isleme, istatistiksel analiz ve modelleme dili |
| **pandas** | Veri okuma (`impacts.csv`, `orbits.csv`), gruplama, yillik agregasyon, tarih donusumu | CSV verisini isleyip yillik kumulatif metrikler (beklenen carpisma, kumulatif olasilik, tehdit eden nesne sayisi) uretmek |
| **numpy** | Yil araliklarinin genisletilmesi, normalizasyon hesaplari | Sayisal islemler ve dizi/aralik uretimi |
| **statsmodels** | `adfuller` (ADF testi), `ARIMA` | Kesif hizi serisinin duragan olup olmadigini test etmek ve p,d,q parametrelerini AIC ile secip 100 yillik projeksiyon uretmek |
| **prophet** (opsiyonel) | Kurulu ise ARIMA yerine otomatik devreye girer | Facebook Prophet ile alternatif/daha esnek trend tahmini |
| **matplotlib** | Trend grafikleri, fark (differencing) grafikleri | Zaman serisinin trend ve durganlik gorsellestirmesi |
| **seaborn** | On yillik Tehlike Isi Haritasi (`tehlike_isi_haritasi.png`) | Hangi on yil / hangi yilin daha riskli oldugunu renk yogunlugu ile gostermek |
| **plotly** | Etkilesimli cizgi grafik (`tehlike_takvimi_cizgi_grafik.html`) | 100 yillik Tehlike Skoru serisini tarayicida yakinlastirilabilir/incelenebilir sekilde sunmak |
| **VS Code Interactive Window** (`# %%` hucre yapisi) | `tehlike_takvimi.py` | Kodun Jupyter benzeri hucreler halinde adim adim calistirilabilmesi |

## Veri Seti Hakkinda Onemli Not

`impacts.csv` (683 satir) ile `orbits.csv` (15.635 satir) arasinda `Object Name` bazinda hicbir ortak kayit bulunmamaktadir (biri provizyonel tasarim kodlu risk listesi nesneleri, digeri numarali/katalogli asteroid yorunge elemanlari). Bu nedenle:

- `orbits.csv`, satir bazinda birlestirilmek yerine tamamlayici baglam (siniflandirma dagilimi, MOID istatistikleri) icin kullanilmistir.
- `impacts.csv` icindeki tum kayitlar zaten gelecege donuk risk pencereleridir (Period Start/End: 2017-2880). Bu yuzden gercek "zaman serisi tahmini" hedefi olarak nesnelerin **kesif yili** (Object Name'deki yil onegi) kullanilarak yillik "yeni risk nesnesi" serisi olusturulmus, ADF/ARIMA ile projekte edilmis ve bilinen risk pencerelerinin dogrudan toplulastirmasiyla birlestirilerek nihai Tehlike Skoru hesaplanmistir.

## Metodoloji (4 Adim)

1. **Kurulum ve Veri Hazirligi** - CSV okuma, `pd.to_datetime` ile tarih standardizasyonu, yillik kumulatif metrik uretimi.
2. **Zaman Serisi Analizi** - Trend/hareketli ortalama incelemesi, Augmented Dickey-Fuller (ADF) testi ile duraganlik kontrolu.
3. **100 Yillik Tehlike Ongorusu** - ADF sonucuna gore secilen ARIMA(p,d,q) modeli (AIC ile otomatik secim) ile kesif hizi projeksiyonu; Prophet kuruluysa alternatif olarak kullanilir.
4. **Etkilesimli Gorsellestirme - Tehlike Takvimi** - Plotly cizgi grafigi ve Seaborn isi haritasi ile en riskli yillarin gorsellestirilmesi.

## Ciktilar

- `tehlike_takvimi.py` - Uctan uca calisan analiz/model kodu (`# %%` hucreleri ile)
- `tehlike_takvimi_2026_2126.csv` - Yil bazinda tum metrikler ve Tehlike Skoru
- `tehlike_takvimi_cizgi_grafik.html` - Etkilesimli Plotly grafigi
- `tehlike_isi_haritasi.png` - On yillik Tehlike Isi Haritasi

## Nerelerde Kullanilabilir

- Uzay/gokbilim farkindaligi ve egitim amacli sunum ve gorsellestirme projelerinde,
- NASA acik verisiyle calisan veri bilimi/zaman serisi portfolyo projelerinde (ornek calisma/case study),
- Planlama ve senaryo analizi gerektiren risk takvimi tarzi diger alanlara (deprem, meteorolojik olay, siber tehdit vb.) uyarlanabilecek bir metodoloji sablonu olarak,
- Zaman serisi tahmin tekniklerinin (ADF testi, ARIMA, opsiyonel Prophet) ogretimi ve gosterimi icin ornek uygulama olarak.

## Nasil Calistirilir

```bash
pip install pandas numpy matplotlib seaborn plotly statsmodels
pip install prophet   # opsiyonel
```

VS Code icinde `tehlike_takvimi.py` dosyasini acip her `# %%` hucresini Interactive Window uzerinden sirayla calistirmak yeterlidir.
