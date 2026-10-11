"""
Module tiền xử lý và làm sạch dữ liệu
Tác giả: Hà Huy Hoàng - 26BCN010
"""
from pathlib import Path
import numpy as np
import pandas as pd


def doc_du_lieu(duong_dan="data/CSDL_TieuDuong.csv"):
    for encoding in ["utf-8-sig", "utf-8", "cp1258", "latin1"]:
        try:
            df = pd.read_csv(duong_dan, encoding=encoding)
            df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
            return df
        except (UnicodeDecodeError, Exception):
            continue
    raise FileNotFoundError(f"Không tìm thấy file: {duong_dan}")


def tien_xu_ly_toan_bo(duong_dan="data/CSDL_TieuDuong.csv"):
    df = doc_du_lieu(duong_dan)

    # 1. Đồng bộ giới tính
    map_gt = {"Male": "Nam", "Female": "Nữ", "male": "Nam", "female": "Nữ"}
    if "GioiTinh" in df.columns:
        df["GioiTinh"] = df["GioiTinh"].replace(map_gt)
        df["GioiTinh_Nam"] = (df["GioiTinh"] == "Nam").astype(int)
        df = df.drop(columns=["GioiTinh"])

    # 2. Đồng bộ nhãn
    map_nhan = {
        "TRUE": 1, "True": 1, "1": 1, 1: 1, 1.0: 1, "Có nguy cơ": 1,
        "FALSE": 0, "False": 0, "0": 0, 0: 0, 0.0: 0, "Không có nguy cơ": 0
    }
    if "NguyCoTieuDuong" in df.columns:
        df["NguyCoTieuDuong"] = df["NguyCoTieuDuong"].map(map_nhan)

    # 3. Ép kiểu số
    cot_so = ["Tuoi", "Glucose", "HbA1C", "Cholesterol", "HDL", "LDL",
              "Triglycerid", "Creatinine", "Ure", "AcidUric", "AST", "ALT"]
    for c in cot_so:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 4. Sửa lỗi mất dấu thập phân
    if "Glucose" in df.columns:
        df.loc[df["Glucose"] > 50, "Glucose"] = df.loc[df["Glucose"] > 50, "Glucose"] / 100
        df.loc[(df["Glucose"] > 20) & (df["Glucose"] < 40), "Glucose"] = df.loc[(df["Glucose"] > 20) & (df["Glucose"] < 40), "Glucose"] / 10
    if "Cholesterol" in df.columns:
        df.loc[df["Cholesterol"] > 20, "Cholesterol"] = df.loc[df["Cholesterol"] > 20, "Cholesterol"] / 10
    if "Creatinine" in df.columns:
        df.loc[df["Creatinine"] < 20, "Creatinine"] = np.nan

    # 5. Xử lý HbA1C điền sẵn 8.025
    if "HbA1C" in df.columns:
        df["HbA1C_DoThuc"] = (df["HbA1C"] != 8.025).astype(int)
        df.loc[df["HbA1C"] == 8.025, "HbA1C"] = np.nan

    # 6. Loại bỏ LDL
    if "LDL" in df.columns:
        df = df.drop(columns=["LDL"])

    # 7. Xóa dòng thiếu nhãn
    df = df.dropna(subset=["NguyCoTieuDuong"]).copy()
    df["NguyCoTieuDuong"] = df["NguyCoTieuDuong"].astype(int)

    return df.reset_index(drop=True)