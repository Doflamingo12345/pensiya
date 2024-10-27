import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import pickle
from plotly_roc import metrics, graphs
from sklearn.metrics import f1_score, confusion_matrix, roc_curve, auc, precision_recall_curve
import plotly.graph_objects as go
import random
from catboost import CatBoostClassifier


from_file = CatBoostClassifier()
from_file.load_model("model")


global df_1
global df_2


st.set_page_config(page_title="Прогнозирование раннего выхода на пенсию",
                           page_icon="⚰️", )
st.title("Аналитика пенсионных данных")

st.sidebar.title("Навигация")
section = st.sidebar.radio("Перейти к разделу",
                           ("Загрузка данных", "Анализ данных"))

# Раздел 1: Загрузка данных
if section == "Загрузка данных":
    st.header("Загрузка данных")

    # Загрузка файлов
    uploaded_file1 = st.file_uploader("Загрузите файл с данными клиентов", type=["csv", "xlsx"])
    uploaded_file2 = st.file_uploader("Загрузите файл с операциями по счетам", type=["csv", "xlsx"])

    if uploaded_file1 and uploaded_file2:
        # Чтение данных
        df_1 = pd.read_csv(uploaded_file1, encoding='cp1251', sep=';') if uploaded_file1.name.endswith('.csv') else pd.read_excel(
            uploaded_file1)
        df_2 = pd.read_csv(uploaded_file2, encoding='cp1251', sep=';') if uploaded_file2.name.endswith('.csv') else pd.read_excel(
            uploaded_file2)
        st.success("Файлы успешно загружены.")
        st.write("Пример данных клиентов:", df_1.head())
        st.write("Пример данных операций:", df_2.head())
    else:
        st.info("Пожалуйста, загрузите оба файла для продолжения.")

# Раздел 2: Анализ данных
elif section == "Анализ данных":
    st.header("Анализ модели на валидационной выборке")

    # Генерация и отображение roc-auc
    pred_df = pd.read_csv("pred_val.csv")
    metrics_df = metrics.metrics_df(pred_df.iloc[:, 0], pred_df.iloc[:, 1])

    st.plotly_chart(
        graphs.roc_curve(
            metrics_df,
            line_name="Pension Prediction",
            line_color="crimson",
            cm_labels=["No Early Retirement", "Early Retirement"],
            fig_size=(600, 600)
        ),
        use_container_width=True
    )
    st.plotly_chart(
        graphs.precision_recall_curve(
            metrics_df,
            line_name="Pension Prediction",
            line_color="crimson",
            cm_labels=["No Early Retirement", "Early Retirement"],
            fig_size=(600, 600)
        ),
        use_container_width=True
    )

