import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# تحميل البيانات
df = pd.read_csv("liver_patient_data.csv")
df = df.dropna()

# تحويل الجنس إلى أرقام
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])

# الميزات والهدف
X = df.drop('Dataset', axis=1)
y = df['Dataset'].apply(lambda x: 1 if x == 1 else 0)

# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# تدريب النموذج
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# حفظ النموذج
joblib.dump(model, 'rf_model.pkl')