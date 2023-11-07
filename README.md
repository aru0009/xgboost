# Импортируем необходимые библиотеки
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Загрузим данные из CSV файла
data = pd.read_csv('popular.csv')
# Подготовим признаки и целевую переменную
X = data[['region', 'year']]  # Признаки (регион и год)
y = data['value']  # Целевая переменная (число людей)
# Преобразуем категориальные признаки в числовой формат
X = pd.get_dummies(X, columns=['region'], drop_first=True)
# Разделим данные на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Создадим и обучим модель XGBoost
model = XGBRegressor()
model.fit(X_train, y_train)
# Предскажем значения на тестовом наборе
y_pred = model.predict(X_test)
# Оценим производительность модели
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R-squared (R2): {r2}')
