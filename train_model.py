import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

df = pd.read_csv("ML/Session 10/PicnicAtBasel.csv")
dl = pd.read_csv("ML/Session 10/PicnicLabel.csv")

df = df.drop(columns=['DATE'])
dl = dl.drop(columns=['DATE'])

print("Dataset Shape:", df.shape)

df_train, df_test, dl_train, dl_test = train_test_split(
    df,
    dl,
    test_size=0.2,
    random_state=42
)

print("Train Size:", df_train.shape)
print("Test Size:", df_test.shape)

pipeline = Pipeline([
    ("scaler",StandardScaler()),
    ("model", RandomForestClassifier(
        n_estimators=100,
        random_state=42))
])

pipeline.fit(df_train, dl_train)

y_pred = pipeline.predict(df_test)

accuracy = accuracy_score(dl_test, y_pred)

print("Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(dl_test, y_pred))

joblib.dump(pipeline, "picnic_model.pkl")

print("\nModel saved as picnic_model.pkl")