"""
Dashboard Prediksi Banjir - Regresi Logistik
Jalankan:  streamlit run app.py
File yang dibutuhkan di folder yang sama: model_terbaik.pkl
"""
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Prediksi Banjir", page_icon="🌧️", layout="wide")

# Konfigurasi fitur (urutan HARUS sama dengan saat pelatihan model)
# label, satuan, min, max, default (median data), step
FITUR = {
    "TN":      ("Suhu minimum", "°C", 15.0, 35.0, 25.8, 0.1),
    "TX":      ("Suhu maksimum", "°C", 20.0, 42.0, 33.9, 0.1),
    "TAVG":    ("Suhu rata-rata", "°C", 20.0, 38.0, 29.1, 0.1),
    "RH_AVG":  ("Kelembapan rata-rata", "%", 0.0, 100.0, 75.0, 1.0),
    "RR":      ("Curah hujan", "mm", 0.0, 300.0, 0.0, 0.1),
    "SS":      ("Lama penyinaran matahari", "jam", 0.0, 12.0, 7.2, 0.1),
    "FF_X":    ("Kecepatan angin maksimum", "m/s", 0.0, 40.0, 6.0, 1.0),
    "DDD_X":   ("Arah angin saat kecepatan maks.", "°", 0.0, 360.0, 150.0, 10.0),
    "FF_AVG":  ("Kecepatan angin rata-rata", "m/s", 0.0, 20.0, 2.0, 1.0),
}
FITUR_P = {k + "_P": v for k, v in FITUR.items()}
# default kolom _P (Stasiun II) sedikit berbeda (median data)
DEFAULT_P = {"TN_P": 25.4, "TX_P": 33.4, "TAVG_P": 28.8, "RH_AVG_P": 75.0, "RR_P": 0.0,
             "SS_P": 6.7, "FF_X_P": 5.0, "DDD_X_P": 120.0, "FF_AVG_P": 2.0}
URUTAN = list(FITUR.keys()) + list(FITUR_P.keys())


@st.cache_resource
def load_model(path="model_terbaik.pkl"):
    return joblib.load(path)


try:
    model = load_model()
except FileNotFoundError:
    st.error("File **model_terbaik.pkl** tidak ditemukan. Letakkan di folder yang sama dengan app.py.")
    st.stop()

kolom_model = list(getattr(model, "feature_names_in_", URUTAN))

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("Pengaturan Treshold")
threshold = st.sidebar.slider(
    "Ambang batas (threshold) probabilitas", 0.05, 0.95, 0.50, 0.05,
    help="Jika probabilitas ≥ ambang batas, hari tersebut diprediksi BANJIR.",
)
st.sidebar.caption(
    "Model dibangun berdasarkan data harian iklim dan banjir di surabaya dengan ±6% hari banjir, sehingga probabilitas model umumnya rendah. "
    "Menurunkan ambang batas (misalnya 0.15–0.25) membuat model lebih peka mendeteksi banjir, "
    "dengan konsekuensi lebih banyak alarm banjir palsu."
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Model:** Regresi Logistik + augmentasi AddNoise (tsaug)")
st.sidebar.markdown(f"**Jumlah fitur:** {len(kolom_model)}")


def label_risiko(p):
    if p >= threshold:
        return "BANJIR", "🔴"
    if p >= threshold * 0.6:
        return "WASPADA", "🟠"
    return "AMAN", "🟢"



# Header
st.title("Dashboard Prediksi Banjir")
st.caption("Prediksi kejadian banjir harian berdasarkan data cuaca menggunakan model regresi logistik yang dibangun berdasarkan data iklim stasiun Maritim Tanjung Perak dan Perak I Surabaya .")

tab1, tab2, tab3 = st.tabs(["Prediksi Manual", "📂 Prediksi dari File", "📊 Interpretasi Model"])

# ---------------------------------------------------------------------------
# TAB 1: Prediksi manual
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("Masukkan data cuaca")
    col_a, col_b = st.columns(2)
    nilai = {}

    with col_a:
        st.markdown("**Data cuaca utama**")
        for k, (lbl, sat, mn, mx, df_, stp) in FITUR.items():
            nilai[k] = st.number_input(f"{lbl} ({sat}) ({k})", mn, mx, df_, stp, key=k)

    with col_b:
        st.markdown("**Data cuaca pembanding (kolom _P)**")
        for k, (lbl, sat, mn, mx, _, stp) in FITUR_P.items():
            nilai[k] = st.number_input(f"{lbl} ({sat}) ({k})", mn, mx, DEFAULT_P[k], stp, key=k)

    if st.button("Prediksi", type="primary"):
        X_in = pd.DataFrame([nilai])[kolom_model]
        prob = float(model.predict_proba(X_in)[0, 1])
        status, ikon = label_risiko(prob)

        st.markdown("---")
        m1, m2, m3 = st.columns(3)
        m1.metric("Probabilitas banjir", f"{prob:.1%}")
        m2.metric("Status", f"{ikon} {status}")
        m3.metric("Ambang batas", f"{threshold:.0%}")
        st.progress(min(prob, 1.0))

        if status == "BANJIR":
            st.error("Kondisi cuaca menunjukkan risiko banjir **tinggi**.")
        elif status == "WASPADA":
            st.warning("Probabilitas mendekati ambang batas, tetap **waspada**.")
        else:
            st.success("Kondisi cuaca menunjukkan risiko banjir **rendah**.")

        # kontribusi tiap fitur terhadap log-odds
        kontribusi = pd.DataFrame({
            "Fitur": kolom_model,
            "Kontribusi (log-odds)": model.coef_[0] * X_in.values[0],
        }).sort_values("Kontribusi (log-odds)", ascending=False)
        with st.expander("Lihat kontribusi fitur terhadap prediksi ini"):
            st.bar_chart(kontribusi.set_index("Fitur"))
            st.caption(f"Intercept model: {model.intercept_[0]:.3f}. "
                       "Nilai positif menaikkan peluang banjir, negatif menurunkan.")

# TAB 2: Prediksi dari file
with tab2:
    st.subheader("Unggah data cuaca (CSV / Excel)")
    st.caption("File harus memuat 18 kolom fitur: " + ", ".join(kolom_model) +
               ". Kolom lain (mis. TANGGAL, Banjir) boleh ada.")
    up = st.file_uploader("Pilih file", type=["csv", "xlsx", "xls"])

    if up is not None:
        df = pd.read_csv(up) if up.name.endswith(".csv") else pd.read_excel(up)
        df.columns = df.columns.astype(str).str.strip()
        hilang = [c for c in kolom_model if c not in df.columns]

        if hilang:
            st.error(f"Kolom berikut tidak ditemukan: {', '.join(hilang)}")
        else:
            data_fitur = df[kolom_model].apply(pd.to_numeric, errors="coerce")
            valid = data_fitur.notna().all(axis=1)
            if (~valid).any():
                st.warning(f"{(~valid).sum()} baris memiliki nilai kosong/non-numerik dan dilewati.")

            hasil = df.loc[valid].copy()
            hasil["Probabilitas"] = model.predict_proba(data_fitur.loc[valid])[:, 1]
            hasil["Prediksi"] = (hasil["Probabilitas"] >= threshold).astype(int)
            hasil["Status"] = hasil["Probabilitas"].apply(lambda p: " ".join(label_risiko(p)[::-1]))

            c1, c2, c3 = st.columns(3)
            c1.metric("Jumlah data", len(hasil))
            c2.metric("Diprediksi banjir", int(hasil["Prediksi"].sum()))
            c3.metric("Rata-rata probabilitas", f"{hasil['Probabilitas'].mean():.1%}")

            # grafik probabilitas (pakai TANGGAL jika ada)
            if "TANGGAL" in hasil.columns:
                grafik = hasil.set_index(pd.to_datetime(hasil["TANGGAL"], errors="coerce"))["Probabilitas"]
            else:
                grafik = hasil["Probabilitas"]
            st.markdown("**Probabilitas banjir per baris**")
            st.line_chart(grafik)

            # evaluasi jika label asli tersedia
            if "Banjir" in hasil.columns:
                y_true = pd.to_numeric(hasil["Banjir"], errors="coerce").fillna(0).astype(int)
                y_pred = hasil["Prediksi"]
                tp = int(((y_true == 1) & (y_pred == 1)).sum())
                fp = int(((y_true == 0) & (y_pred == 1)).sum())
                fn = int(((y_true == 1) & (y_pred == 0)).sum())
                tn = int(((y_true == 0) & (y_pred == 0)).sum())
                prec = tp / (tp + fp) if tp + fp else 0
                rec = tp / (tp + fn) if tp + fn else 0
                f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0

                st.markdown("**Evaluasi (kolom Banjir terdeteksi)**")
                e1, e2, e3, e4 = st.columns(4)
                e1.metric("Akurasi", f"{(tp + tn) / len(y_true):.1%}")
                e2.metric("Precision", f"{prec:.1%}")
                e3.metric("Recall", f"{rec:.1%}")
                e4.metric("F1-score", f"{f1:.3f}")
                st.table(pd.DataFrame(
                    [[tn, fp], [fn, tp]],
                    index=["Aktual: Tidak banjir", "Aktual: Banjir"],
                    columns=["Prediksi: Tidak banjir", "Prediksi: Banjir"],
                ))

            st.markdown("**Tabel hasil**")
            st.dataframe(hasil)
            st.download_button(
                "⬇️ Unduh hasil prediksi (CSV)",
                hasil.to_csv(index=False).encode("utf-8"),
                file_name="hasil_prediksi_banjir.csv",
                mime="text/csv",
            )


# TAB 3: Interpretasi model (dihitung langsung dari model_terbaik.pkl)
with tab3:
    st.subheader("Odds ratio tiap fitur")
    odds = pd.DataFrame({
        "Fitur": kolom_model,
        "Koefisien": model.coef_[0],
        "Odds Ratio": np.exp(model.coef_[0]),
    })
    odds["Pengaruh"] = np.where(odds["Odds Ratio"] > 1, "⬆️ Menaikkan risiko", "⬇️ Menurunkan risiko")
    odds = odds.sort_values("Odds Ratio", ascending=False).reset_index(drop=True)

    st.bar_chart(odds.set_index("Fitur")["Odds Ratio"] - 1)
    st.caption("Grafik menampilkan (Odds Ratio dari model terbaik yang telah ditraining sebelumnya). Odd rasio positif = fitur menaikkan peluang banjir.")
    st.dataframe(odds.style.format({"Koefisien": "{:.4f}", "Odds Ratio": "{:.4f}"}))
    st.info(
        "**Cara membaca:** odds ratio 1,05 berarti setiap kenaikan 1 satuan pada fitur tersebut "
        "meningkatkan peluang (odds) banjir sekitar 5%, dengan fitur lain dianggap tetap."
    )
