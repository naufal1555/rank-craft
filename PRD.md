# Product Requirements Document (PRD)
## RankCraft: Counterfactual Learning-to-Rank (CLTR) & Position-Debiased RecSys Engine for Traveloka

**Version**: 1.0.0  
**Author**: Ahmad Fathin Naufal (`naufal1555`)  
**Target Domain**: Traveloka / Tiket.com Search Ranking, Recommendation & Discovery Data Science  
**Status**: Approved & In Execution  

---

## 1. Problem Statement & Business Context
Pada platform Online Travel Agent (OTA) seperti Traveloka (pencarian Hotel dan Tiket Pesawat), sebagian besar konversi berasal dari halaman hasil pencarian (*Search Engine Results Page* / SERP).

### Cacat Fundamental Model Learning-to-Rank Konvensional:
Model peringkat standar (seperti XGBoost/LightGBM LambdaMART yang dilatih langsung pada log klik pengguna) menderita **Position Bias** yang parah:
1. **Visual Saliency Confounding**: Pengguna mengklik item di peringkat #1-#3 bukan semata-mata karena item tersebut paling relevan, melainkan karena item tersebut paling pertama dilihat (*visual prominence*).
2. **Rich-Get-Richer Feedback Loops**: Hotel yang kebetulan berada di urutan teratas mendapatkan banyak klik, sehingga model naive menganggap hotel tersebut semakin relevan, menguncinya di peringkat atas. Sementara hotel baru atau hotel bintang lima bernilai tinggi di peringkat bawah tidak pernah dilihat pengguna, memicu *cold-start starvation*.
3. **Penyimpangan Evaluasi Offline**: Model yang dilatih pada data berbias tampak memiliki NDCG offline tinggi, namun gagal meningkatkan konversi dan GMV saat diuji dalam A/B test online.

**RankCraft** memecahkan masalah ini dengan menerapkan **Counterfactual Learning-to-Rank (CLTR)**:
Mengestimasi kecenderungan pemeriksaan pengguna (*User Examination Propensity* $p_k$) dan mengoreksi bias posisi menggunakan **Inverse Propensity Scoring (IPS)** dan **Doubly Robust (DR)** Estimators (Joachims et al. KDD 2017), dipadukan dengan **Contextual Thompson Sampling** untuk eksplorasi inventaris hotel baru.

---

## 2. Success Metrics & Key Performance Indicators (KPIs)

| Kategori Metrik | KPI | Target Minimal | Definisi Matematis / Operasional |
| :--- | :--- | :--- | :--- |
| **Ranking Accuracy** | **Unbiased NDCG@10** | $\ge 0.820$ | $\text{NDCG}@K$ yang dihitung terhadap *true latent relevance* (ground truth utilitas). Target: lift $+15\%$ dibanding model Naive. |
| **Top-Slot Precision** | **Unbiased NDCG@5** | $\ge 0.850$ | Kualitas relevansi pada 5 slot teratas (layar pertama mobile app). |
| **Exploration Quality** | **Cold-Start Discovery Rate** | $\ge 25\%$ uplift | Peningkatan frekuensi keterpajangan hotel baru berkualitas tinggi tanpa mengorbankan GMV. |
| **Bias Mitigation** | **Rank Correlation (Kendall's $\tau$)** | $\ge 0.70$ | Korelasi peringkat model terhadap urutan utilitas sejati pengguna. |
| **Serving Performance** | **Real-Time Latency SLA** | $p_{95} < 15$ ms | Inferensi scoring reranking per request pencarian (50 kandidat per query). |

---

## 3. Mathematical & Theoretical Framework

### 3.1. User Click Behavior Model
Mengacu pada **Position-Based Click Model (PBM)**:
$$\mathbb{P}(C_i = 1 \mid x_i, k_i) = \mathbb{P}(E_i = 1 \mid k_i) \cdot \mathbb{P}(R_i = 1 \mid x_i)$$
- $C_i \in \{0, 1\}$: Kejadian pengguna mengklik hotel $i$.
- $E_i \in \{0, 1\}$: Variabel laten apakah pengguna memeriksa/melihat (*examine*) slot peringkat $k_i$.
- $R_i \in \{0, 1\}$: Variabel laten apakah hotel $i$ benar-benar relevan dengan preferensi pengguna $x_i$.
- $p_k = \mathbb{P}(E = 1 \mid k)$: Probabilitas pemeriksaan posisi $k$, dimodelkan dengan *power-law decay*:
  $$p_k = \left(\frac{1}{k}\right)^\gamma, \quad \gamma \in [0.5, 0.8]$$

### 3.2. Inverse Propensity Scoring (IPS) Estimator
Untuk mengeliminasi bias posisi, setiap observasi klik $C_{q,i}$ diboboti dengan kebalikan dari probabilitas pemeriksaan:
$$\mathcal{L}_{\text{IPS}}(\theta) = \sum_{q \in Q} \sum_{i \in q} \frac{C_{q,i}}{p_{k_i}} \cdot \ell\left(f_\theta(x_{q,i}), y_{q,i}^*\right)$$
*Sifat*: Estimator IPS bersifat **unbiased**:
$$\mathbb{E}_{E}\left[ \frac{C_i}{p_{k_i}} \right] = \frac{\mathbb{E}[C_i]}{p_{k_i}} = \frac{p_{k_i} \cdot R_i}{p_{k_i}} = R_i$$

### 3.3. Doubly Robust (DR) Estimator
Estimator IPS murni dapat memiliki varians tinggi jika $p_k$ sangat kecil di posisi bawah. RankCraft mengimplementasikan estimator **Doubly Robust (DR)**:
$$\mathcal{L}_{\text{DR}}(\theta) = \sum_{q} \sum_{i} \left[ \hat{r}_{q,i} + \frac{C_{q,i} - \hat{r}_{q,i} \cdot p_{k_i}}{p_{k_i}} \right] \cdot \ell\left(f_\theta(x_{q,i})\right)$$
Di mana $\hat{r}_{q,i}$ adalah model imputasi baseline relevansi. Estimator DR terbukti *unbiased* jika salah satu dari model propensitas atau model imputasi teridentifikasi dengan benar.

### 3.4. Contextual Thompson Sampling for Exploration
Untuk hotel baru (*cold-start*), skor akhir menggabungkan eksploitasi debiased ranker dengan pengambilan sampel stokastik Bayes dari distribusi posterior Beta $(\alpha_i, \beta_i)$:
$$\text{Score}(i) = f_{\text{DR}}(x_i) + \lambda \cdot \theta_i, \quad \theta_i \sim \text{Beta}(\alpha_i, \beta_i)$$

---

## 4. Dataset & Query Simulation (Traveloka Hotel Search)

Fitur-fitur pencarian yang dimodelkan:
- **Query Context**: Lokasi tujuan (Bali, Jakarta, Yogyakarta, Bandung, Surabaya), Tanggal menginap, Jumlah tamu, Kategori anggaran (Budget, Midscale, Luxury).
- **Hotel Features**: Harga per malam (IDR), Rating bintang (1-5), Skor review pengguna (1.0-10.0), Jarak ke pusat kota / pantai (km), Diskon promo (%), Jumlah fasilitas (kolam renang, WiFi, sarapan).
- **Historical Telemetry**: Historical CTR, Conversion rate, Tingkat pembatalan.
- **SERP Position Log**: Posisi tayang ($1 - 20$), Status klik observasi, True latent relevance.

---

## 5. Execution Workflow
1. **Dataset Synthesis** (`src/dataset.py`): 50.000 log pencarian hotel dengan bias posisi mata *eye-tracking* dan utilitas laten sejati.
2. **CLTR Engine** (`src/models.py`): Implementasi Naive Ranker, IPS Ranker, Doubly Robust Ranker, dan Contextual Thompson Sampling Bandit.
3. **Benchmarking** (`src/evaluate.py`): Evaluasi Unbiased NDCG@5, NDCG@10, Expected Clicks, dan plot perbandingan.
4. **FastAPI Serving** (`src/api.py`): Microservice reranking real-time dengan latensi $< 10$ ms.
5. **Interactive Dashboard** (`dashboard/app.py`): Streamlit SERP simulator membandingkan visualisasi hasil pencarian Biased vs Debiased.
6. **Git CI & GitHub Publishing**: Repo publik `naufal1555/rank-craft` dengan kontributor tunggal bersih.
