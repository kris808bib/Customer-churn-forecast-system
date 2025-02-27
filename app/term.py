import pickle
import pandas as pd
import numpy as np

def process_df(df):
    
    # Создаем копию DataFrame
    df_processed = df.copy()
    
    # Фильтрация столбцов, в которых необходимо заменить значения
    filtered_columns_for_replace = df_processed.columns[df_processed.nunique() == 3]
    for col in filtered_columns_for_replace:
        df_processed[col] = df_processed[col].apply(lambda x: 'No' if 'No ' in str(x) else x)

    # Фильтрация столбцов для факторизации
    filtered_columns_for_factorize = df_processed.columns[df_processed.nunique() <= 4]
    for col in filtered_columns_for_factorize:
        df_processed[col] = pd.factorize(df_processed[col])[0]

    # Удаление столбца customerID, если он присутствует
    if 'customerID' in df_processed.columns:
        df_processed = df_processed.drop(['customerID'], axis=1)

    # Преобразование TotalCharges в числовой тип и заполнение пропусков
    df_processed['TotalCharges'] = pd.to_numeric(df_processed['TotalCharges'], errors='coerce')
    df_processed['TotalCharges'] = df_processed['TotalCharges'].fillna(df_processed['TotalCharges'].mean(), inplace=True)

    # Создание новых признаков
    df_processed['MonthlyCharges_per_tenure'] = df_processed['MonthlyCharges'] / df_processed['tenure'].replace(0, 1)  
    df_processed['TotalCharges_per_Month'] = df_processed['TotalCharges'] / df_processed['tenure'].replace(0, 1)  

    # Подсчет количества услуг, если передан список
    services =['PhoneService', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    df_processed['Service_Count'] = df_processed[services].sum(axis=1)

    # Создание признака Has_Streaming
    df_processed['Has_Streaming'] = ((df_processed['StreamingTV'] == 1) | (df_processed['StreamingMovies'] == 1)).astype(int)

    # Создание дополнительных признаков
    df_processed['Tenure_Squared'] = df_processed['tenure'] ** 2
    df_processed['Tenure_Log'] = np.log(df_processed['tenure'] + 1)  

    # Замена inf на NaN и заполнение NaN средними значениями
    df_processed.replace([np.inf, -np.inf], np.nan, inplace=True)  
    df_processed.fillna(df_processed.mean(), inplace=True)

    # Удаление указанных столбцов, если они есть
    drop_columns=['gender', 'PhoneService']
    df_processed = df_processed.drop(columns=drop_columns, errors='ignore')

    return df_processed


# Функция для определения категории риска
def get_risk_category(probability, thresholds=[0.3, 0.7]):
    if probability < thresholds[0]:
        return 'Низкий риск'
    elif probability < thresholds[1]:
        return 'Средний риск'
    else:
        return 'Высокий риск'

# Функция для генерации рекомендаций
def generate_recommendations(risk_category, client_data):
    recommendations = []
    
    if risk_category == 'Низкий риск':
        recommendations.append("Продолжайте использовать текущие услуги.")
        
    elif risk_category == 'Средний риск':
        recommendations.append("Рассмотрите возможность подключения дополнительных услуг.")
        
        if client_data['OnlineSecurity'] == 0:
            recommendations.append("Рекомендуем подключить услугу онлайн-защиты.")
        if client_data['OnlineBackup'] == 0:
            recommendations.append("Рекомендуем подключить услугу онлайн-резервного копирования.")
        if client_data['DeviceProtection'] == 0:
            recommendations.append("Рекомендуем подключить услугу защиты устройств.")
        if client_data['TechSupport'] == 0:
            recommendations.append("Рекомендуем подключить услугу технической поддержки.")
        if client_data['StreamingTV'] == 0 or client_data['StreamingMovies'] == 0:
            recommendations.append("Рекомендуем подключить услуги стриминга ТВ или фильмов.")
        
    elif risk_category == 'Высокий риск':
        recommendations.append("Свяжитесь с клиентом для уточнения причин недовольства.")
        
        if client_data['Contract'] == 'Month-to-month':
            recommendations.append("Предложите переход на долгосрочный контракт со скидкой.")
        if client_data['MonthlyCharges'] > 70:
            recommendations.append("Предложите специальный тарифный план.")
        if client_data['Service_Count'] < 3:
            recommendations.append("Предложите дополнительные услуги для повышения ценности обслуживания.")
        if client_data['Has_Streaming'] == 0:
            recommendations.append("Рекомендуем подключить стриминговые услуги для улучшения клиентского опыта.")
    return "; ".join(recommendations)

# Загрузка тестового датасета
test_dataset = pd.read_csv('D:\\Прогноз оттока\\app\data\\test_dataset.csv')

# Обработка тестового датасета
test_dataset = process_df(test_dataset)

# Загрузка модели из файла
try:
    with open('D:\\Прогноз оттока\\app\\models\\model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    raise FileNotFoundError("Файл модели 'model.pkl' не найден.")

# Проверка, что модель загружена
if not hasattr(model, 'predict'):
    raise ValueError("Модель не загружена или не обучена.")

# Применение модели к тестовому датасету
for index, row in test_dataset.iterrows():
    client_data = row.to_dict()
    client_df = pd.DataFrame([client_data])

    # Предсказание вероятности оттока
    churn_probability = model.predict_proba(client_df)[:, 1][0]
    
    # Определение категории риска
    risk_category = get_risk_category(churn_probability)
    
    # Генерация рекомендаций
    recommendations = generate_recommendations(risk_category, client_data)
    
    # Вывод результатов
    # print(f"Клиент {index + 1}:")  # Используем индекс, так как customerID удален
    # print(f"  Вероятность оттока: {churn_probability:.2f}")
    # print(f"  Категория риска: {risk_category}")
    # print(f"  Рекомендации: {recommendations}")
    # print("-" * 50)