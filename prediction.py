import joblib


def predict(data):
    xgb_model = joblib.load("xgb_model.sav")
    return xgb_model.predict(data)
