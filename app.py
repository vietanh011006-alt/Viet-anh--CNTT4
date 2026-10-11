import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Du doan nguy co tieu duong", page_icon="🩺")

st.title("🩺 Du doan Nguy co Benh Tieu Duong")
st.caption("Sinh vien: Ha Huy Hoang - 26BCN010")
st.caption("Phan mem chi ho tro sang loc, khong thay the chan doan bac si.")

# ======== TAI MO HINH ========
from pathlib import Path

THU_MUC_DU_AN = Path(__file__).resolve().parent
DUONG_DAN_MO_HINH = THU_MUC_DU_AN / "outputs" / "mo_hinh_tot_nhat.joblib"


@st.cache_resource
def tai_mo_hinh():
    if not DUONG_DAN_MO_HINH.exists():
        st.error(f"Không tìm thấy mô hình tại: {DUONG_DAN_MO_HINH}")
        st.stop()

    return joblib.load(DUONG_DAN_MO_HINH)

model = tai_mo_hinh()

# ======== NHAP LIEU ========
st.header("📋 Nhap thong tin xet nghiem")

col1, col2 = st.columns(2)

with col1:
    ho_ten = st.text_input("Ho ten", value="Benh nhan")
    tuoi = st.number_input("Tuoi", min_value=1, max_value=120, value=40)
    gioi_tinh = st.selectbox("Gioi tinh", ["Nam", "Nu"])
    glucose = st.number_input("Glucose (mmol/L)", min_value=1.0, max_value=50.0, value=5.5, step=0.1)
    cholesterol = st.number_input("Cholesterol (mmol/L)", min_value=1.0, max_value=20.0, value=5.0, step=0.1)
    hdl = st.number_input("HDL (mmol/L)", min_value=0.1, max_value=5.0, value=1.5, step=0.1)
    triglycerid = st.number_input("Triglycerid (mmol/L)", min_value=0.1, max_value=20.0, value=2.0, step=0.1)

with col2:
    creatinine = st.number_input("Creatinine (umol/L)", min_value=10.0, max_value=1500.0, value=80.0, step=0.1)
    ure = st.number_input("Ure (mmol/L)", min_value=1.0, max_value=50.0, value=5.5, step=0.1)
    acid_uric = st.number_input("Acid uric (umol/L)", min_value=50.0, max_value=1000.0, value=350.0, step=0.1)
    ast = st.number_input("AST (U/L)", min_value=1.0, max_value=1000.0, value=25.0, step=0.1)
    alt = st.number_input("ALT (U/L)", min_value=1.0, max_value=1000.0, value=25.0, step=0.1)

# HbA1C khong bat buoc
co_hba1c = st.checkbox("Co xet nghiem HbA1C")
if co_hba1c:
    hba1c = st.number_input("HbA1C (%)", min_value=3.0, max_value=20.0, value=6.0, step=0.1)
    hba1c_do_thuc = 1
else:
    hba1c = 8.025  # gia tri dien san (sau do de NaN trong mo hinh)
    hba1c_do_thuc = 0

# ======== DU DOAN ========
if st.button("🔮 Dự đoán", type="primary"):

    # Tao DataFrame cho mo hinh
    input_data = pd.DataFrame([{
        "Tuoi": tuoi,
        "GioiTinh_Nam": 1 if gioi_tinh == "Nam" else 0,
        "Glucose": glucose,
        "HbA1C": np.nan if hba1c_do_thuc == 0 else hba1c,
        "HbA1C_DoThuc": hba1c_do_thuc,
        "Cholesterol": cholesterol,
        "HDL": hdl,
        "Triglycerid": triglycerid,
        "Creatinine": creatinine,
        "Ure": ure,
        "AcidUric": acid_uric,
        "AST": ast,
        "ALT": alt
    }])

    # Du doan
    xac_suat = model.predict_proba(input_data)[0, 1]
    nhan = model.predict(input_data)[0]

    # ======== LUAT CHUYEN GIA (ADA) ========
    canh_bao_theo_luat = "Thap"
    if glucose >= 7.0 or (co_hba1c and hba1c >= 6.5):
        canh_bao_theo_luat = "Cao"
    elif glucose >= 6.1 or (co_hba1c and hba1c >= 5.7):
        canh_bao_theo_luat = "Trung binh"

    # Muc canh bao tu mo hinh
    if xac_suat >= 0.7:
        canh_bao_mo_hinh = "Cao"
    elif xac_suat >= 0.3:
        canh_bao_mo_hinh = "Trung binh"
    else:
        canh_bao_mo_hinh = "Thap"

    # Lay muc cao hon giua 2 nguon
    muc_do = {"Thap": 0, "Trung binh": 1, "Cao": 2}
    muc_cuoi = max(muc_do[canh_bao_mo_hinh], muc_do[canh_bao_theo_luat])
    ten_muc = {0: "Thap", 1: "Trung binh", 2: "Cao"}

    # ======== HIEN THI KET QUA ========
    st.header("📊 Kết quả dự đoán")

    col_a, col_b = st.columns(2)

    with col_a:
        st.metric("Xac suat nguy co", f"{xac_suat*100:.1f}%")
    with col_b:
        st.metric("Muc canh bao", ten_muc[muc_cuoi])

    if muc_cuoi == 2:
        st.error("🔴 **NGUY CO CAO** - Ban nen den co so y te xet nghiem khang dinh som.")
    elif muc_cuoi == 1:
        st.warning("🟠 **NGUY CO TRUNG BINH** - Ban nen theo doi, xet nghiem lai sau 1-3 thang, dieu chinh loi song.")
    else:
        st.success("🟢 **NGUY CO THAP** - Duy tri loi song lanh manh, kham dinh ky.")

    # ======== GIAI THICH TUNG CHI SO ========
    st.header("🔍 Giải thích chi tiết")

    chi_so = [
        ("Glucose", glucose, 3.9, 6.1, "mmol/L"),
        ("Cholesterol", cholesterol, 2.8, 5.2, "mmol/L"),
        ("HDL", hdl, 1.0, 1.6, "mmol/L"),
        ("Triglycerid", triglycerid, 0.4, 1.7, "mmol/L"),
        ("Creatinine", creatinine, 44, 106, "umol/L"),
        ("Ure", ure, 2.8, 7.2, "mmol/L"),
        ("AcidUric", acid_uric, 210, 420, "umol/L"),
        ("AST", ast, 10, 40, "U/L"),
        ("ALT", alt, 7, 56, "U/L"),
    ]

    for ten, gia_tri, thap, cao, don_vi in chi_so:
        if gia_tri < thap:
            st.write(f"  ⬇️ **{ten}**: {gia_tri} {don_vi} (thap hon binh thuong {thap})")
        elif gia_tri > cao:
            st.write(f"  ⬆️ **{ten}**: {gia_tri} {don_vi} (cao hon binh thuong {cao})")
        else:
            st.write(f"  ✅ **{ten}**: {gia_tri} {don_vi} (binh thuong)")

    if co_hba1c:
        if hba1c >= 6.5:
            st.write(f"  ⬆️ **HbA1C**: {hba1c}% (cao, >= 6.5% = tieu duong)")
        elif hba1c >= 5.7:
            st.write(f"  ⬆️ **HbA1C**: {hba1c}% (trung binh, 5.7-6.4% = tien tieu duong)")
        else:
            st.write(f"  ✅ **HbA1C**: {hba1c}% (binh thuong)")

    # ======== TOM TAT ========
    st.header("📌 Tom tat")
    nhanh_chinh = []
    if glucose >= 7.0:
        nhanh_chinh.append(f"Glucose {glucose} mmol/L vuot nguong tieu duong (7.0)")
    elif glucose >= 6.1:
        nhanh_chinh.append(f"Glucose {glucose} mmol/L o muc tien tieu duong (6.1-6.9)")
    if co_hba1c and hba1c >= 6.5:
        nhanh_chinh.append(f"HbA1C {hba1c}% vuot nguong tieu duong (6.5%)")

    if nhanh_chinh:
        for item in nhanh_chinh:
            st.write(f"  • **{item}**")
    else:
        st.write("  • Cac chi so duong huyet deu o muc binh thuong.")

    st.caption("⚠️ Phan mem nay chi co gia tri ho tro sang loc, khong thay the chan doan cua bac si.")