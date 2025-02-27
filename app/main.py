from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
from term import process_df, get_risk_category, generate_recommendations
import pickle
import os

app = FastAPI()

# Указываем папку с шаблонами
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))

# Загрузка модели и данных
current_dir = os.path.dirname(os.path.abspath(__file__))
test_dataset_path = os.path.join(current_dir, "data", "test_dataset.csv")

model_path = os.path.join(current_dir, "models", "model.pkl")

test_dataset = pd.read_csv(test_dataset_path)
test_dataset = process_df(test_dataset)
with open(model_path, "rb") as file:
    model = pickle.load(file)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.get("/client/{client_id}", response_class=HTMLResponse)
async def get_client_recommendations(request: Request, client_id: int):
    client_data = test_dataset.iloc[client_id].to_dict()
    client_df = pd.DataFrame([client_data])
    churn_probability = model.predict_proba(client_df)[:, 1][0]
    risk_category = get_risk_category(churn_probability)
    recommendations = generate_recommendations(risk_category, client_data)
    return templates.TemplateResponse(
        "client.html",
        {
            "request": request,
            "client_id": client_id,
            "churn_probability": f"{churn_probability:.2f}",
            "risk_category": risk_category,
            "recommendations": recommendations,
        },
    )

# @app.get("/client/{client_id}", response_class=HTMLResponse)
# async def get_client_recommendations(request: Request, client_id: int):
#     # Получаем данные клиента
#     client_data = test_dataset.iloc[client_id].to_dict()
#     print(f"\nClient {client_id} raw data: {client_data}")  # Отладочный вывод

#     # Обрабатываем данные
#     client_df = pd.DataFrame([client_data])
#     print(f"Processed data for client {client_id}:\n{client_df}")  # Отладочный вывод

#     # Проверяем, что данные после обработки различаются
#     if client_id > 0:
#         prev_client_data = test_dataset.iloc[client_id - 1].to_dict()
#         prev_client_df = process_df(pd.DataFrame([prev_client_data]))
#         if client_df.equals(prev_client_df):
#             print(f"WARNING: Processed data for client {client_id} is the same as for client {client_id - 1}!")

#     # Предсказание вероятности оттока
#     churn_probability = model.predict_proba(client_df)[:, 1][0]
#     print(f"Churn probability for client {client_id}: {churn_probability:.2f}")  # Отладочный вывод

#     # Определение категории риска
#     risk_category = get_risk_category(churn_probability)
#     print(f"Risk category for client {client_id}: {risk_category}")  # Отладочный вывод

#     # Генерация рекомендаций
#     recommendations = generate_recommendations(risk_category, client_data)
#     print(f"Recommendations for client {client_id}: {recommendations}")  # Отладочный вывод

#     return templates.TemplateResponse(
#         "client.html",
#         {
#             "request": request,
#             "client_id": client_id,
#             "churn_probability": f"{churn_probability:.2f}",
#             "risk_category": risk_category,
#             "recommendations": recommendations,
#         },
#     )

@app.get("/clients", response_class=HTMLResponse)
async def get_clients_list(request: Request):
    results = []
    for client_id in range(len(test_dataset)):
        client_data = test_dataset.iloc[client_id].to_dict()
        client_df = pd.DataFrame([client_data])
        churn_probability = model.predict_proba(client_df)[:, 1][0]
        risk_category = get_risk_category(churn_probability)
        recommendations = generate_recommendations(risk_category, client_data)
        results.append({
            "client_id": client_id,
            "churn_probability": f"{churn_probability:.2f}",
            "risk_category": risk_category,
            "recommendations": recommendations,
        })
    results.sort(key=lambda x: float(x["churn_probability"]), reverse=True)
    return templates.TemplateResponse(
        "clients.html",
        {"request": request, "results": results},
    )

@app.get("/general_recommendations", response_class=HTMLResponse)
async def get_general_recommendations(request: Request):
    recommendations = []
    for client_id in range(len(test_dataset)):
        client_data = test_dataset.iloc[client_id].to_dict()
        client_df = pd.DataFrame([client_data])
        
        churn_probability = model.predict_proba(client_df)[:, 1][0]
        risk_category = get_risk_category(churn_probability)
        client_recommendations = generate_recommendations(risk_category, client_data)
        recommendations.append({
            "client_id": client_id,
            "recommendations": client_recommendations,
        })
    return templates.TemplateResponse(
        "general_recommendations.html",
        {"request": request, "recommendations": recommendations},
    )