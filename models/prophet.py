from prophet import Prophet
import pandas as pd
import numpy as np


class MyProphet:
    def __init__(self, args):
        self.response_col = args.response_col
        self.date_col = args.date_col

    def fit(self, data_x):
        # إنشاء نموذج Prophet
        def make_model():
            return Prophet(
                daily_seasonality=False,
                weekly_seasonality=True,
                yearly_seasonality=True
            )

        self.model_fbp = make_model()
        self.regressors = []

        # تحديد الأعمدة المستخدمة كـ regressors
        for col in data_x.columns:
            if col != self.response_col and col != self.date_col:
                self.regressors.append(col)

        # تنظيف وتحويل البيانات الرقمية فقط
        data_x = data_x.copy()
        for r in self.regressors:
            data_x[r] = pd.to_numeric(data_x[r], errors="coerce")
        data_x[self.response_col] = pd.to_numeric(data_x[self.response_col], errors="coerce")

        # حذف القيم الفارغة
        data_x = data_x.dropna(subset=[self.date_col, self.response_col]).reset_index(drop=True)

        # إعادة تسمية الأعمدة لتناسب Prophet
        ml_df1 = data_x.rename(columns={self.date_col: "ds", self.response_col: "y"})

        # التأكد من أن البيانات كافية
        if len(ml_df1) < 2:
            print("⚠️ Not enough data after cleaning. Please check your dataset.")
            return

        # ✅ تدريب آمن على Windows باستخدام LBFGS فقط
        try:
            self.model_fbp.fit(ml_df1, algorithm="LBFGS", iter=2000)
        except Exception as e:
            print(f"⚠️ Prophet encountered an error: {e}")
            print("Retrying with a new model and fewer iterations...")
            # إعادة إنشاء النموذج وإعادة التدريب
            self.model_fbp = make_model()
            try:
                self.model_fbp.fit(ml_df1, algorithm="LBFGS", iter=500)
            except Exception as e2:
                print(f"❌ Prophet failed again: {e2}")
                self.model_fbp = None

    def predict(self, test_x):
        # إذا النموذج فشل بالتدريب نوقف
        if self.model_fbp is None:
            print("⚠️ Model is not trained. Cannot predict.")
            return pd.Series([])

        # تجهيز بيانات التنبؤ
        test_x = test_x.copy()
        for r in self.regressors:
            if r in test_x.columns:
                test_x[r] = pd.to_numeric(test_x[r], errors="coerce")

        test_x = test_x.rename(columns={self.date_col: "ds", self.response_col: "y"})
        test_x = test_x.dropna().reset_index(drop=True)

        if len(test_x) == 0:
            print("⚠️ No valid rows in test data for prediction.")
            return pd.Series([])

        # تنفيذ التنبؤ
        pred_y = self.model_fbp.predict(test_x)
        return pred_y["yhat"]