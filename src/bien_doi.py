"""
Module biến đổi dữ liệu (Winsorizing, Imputation, Scaling)
Tác giả: Hà Huy Hoàng - 26BCN010
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class BoXuLyThieu(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.median_dict = {}

    def fit(self, X, y=None):
        df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        for col in df.columns:
            if df[col].dtype in ["float64", "float32", "int64", "int32"]:
                self.median_dict[col] = df[col].median()
        return self

    def transform(self, X):
        df = pd.DataFrame(X.copy()) if not isinstance(X, pd.DataFrame) else X.copy()
        for col in df.columns:
            if col in self.median_dict:
                df[col] = df[col].fillna(self.median_dict[col])
        return df


class BoCatNgoaiLai(BaseEstimator, TransformerMixin):
    def __init__(self, phan_vi_duoi=1, phan_vi_tren=99):
        self.phan_vi_duoi = phan_vi_duoi
        self.phan_vi_tren = phan_vi_tren
        self.nguong = {}

    def fit(self, X, y=None):
        df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        for col in df.columns:
            if df[col].dtype in ["float64", "float32", "int64", "int32"]:
                d = np.percentile(df[col].dropna(), self.phan_vi_duoi)
                t = np.percentile(df[col].dropna(), self.phan_vi_tren)
                if d < t:
                    self.nguong[col] = (d, t)
        return self

    def transform(self, X):
        df = pd.DataFrame(X.copy()) if not isinstance(X, pd.DataFrame) else X.copy()
        for col, (d, t) in self.nguong.items():
            if col in df.columns:
                df[col] = df[col].clip(lower=d, upper=t)
        return df


class BoChuanHoa(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.mean_dict = {}
        self.std_dict = {}

    def fit(self, X, y=None):
        df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        for col in df.columns:
            if df[col].dtype in ["float64", "float32", "int64", "int32"]:
                self.mean_dict[col] = df[col].mean()
                self.std_dict[col] = df[col].std() if df[col].std() != 0 else 1.0
        return self

    def transform(self, X):
        df = pd.DataFrame(X.copy()) if not isinstance(X, pd.DataFrame) else X.copy()
        for col in self.mean_dict:
            if col in df.columns:
                df[col] = (df[col] - self.mean_dict[col]) / self.std_dict[col]
        return df


class PipelineBienDoi:
    def __init__(self):
        self.bo_thieu = BoXuLyThieu()
        self.bo_cat = BoCatNgoaiLai()
        self.bo_chuan_hoa = BoChuanHoa()

    def fit_transform(self, X):
        X1 = self.bo_thieu.fit_transform(X)
        X2 = self.bo_cat.fit_transform(X1)
        X3 = self.bo_chuan_hoa.fit_transform(X2)
        return X3

    def transform(self, X):
        X1 = self.bo_thieu.transform(X)
        X2 = self.bo_cat.transform(X1)
        X3 = self.bo_chuan_hoa.transform(X2)
        return X3