import pandas as pd
import numpy as np
import streamlit as st
from tqdm import tqdm # Библиотека для визуализации прогресса циклов
from plotly_roc import metrics, graphs
import matplotlib.pyplot as plt
import seaborn as sns # Библиотеки для визуализации

from sklearn.preprocessing import LabelEncoder, StandardScaler # Инструменты для предобработки данных
from sklearn.model_selection import train_test_split # Разделение данных на обучающую и тестовую выборки
import scipy.stats as stats # Статистические инструменты

from sklearn.metrics import f1_score, confusion_matrix, roc_curve, auc # Метрики для оценки модели
import os # Работа с файловой системой

import torch
from torch.utils.data import Dataset, DataLoader # Инструменты PyTorch для работы с данными

from catboost import CatBoostClassifier # Модель CatBoost для классификации

df_1 = pd.DataFrame()
df_2 = pd.DataFrame()



st.set_page_config(page_title="Прогнозирование раннего выхода на пенсию",
                           page_icon="⚰️", )
st.title("Аналитика пенсионных данных")

st.sidebar.title("Навигация")
# Раздел 1: Загрузка данных
section = st.sidebar.radio("Перейти к разделу",
                           ("Загрузка данных", "Анализ данных"))
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
if section == "Анализ данных":
    st.header("Анализ модели на валидационной выборке")
    # Настройка каталога для сохранения обработанных батчей
    output_dir = "processed_batches"
    os.makedirs(output_dir, exist_ok=True)

    tr_dummies_1 = pd.get_dummies(df_2[['cmmnt', 'sum_type', 'mvmnt_type']])
    # Удаление исходных категориальных столбцов из df_2 и добавление фиктивных переменных
    df_2 = df_2.drop(['cmmnt', 'sum_type', 'mvmnt_type'], axis=1).join(tr_dummies_1)
    # Группировка данных в df_2 по accnt_id и вычисление суммы по каждому столбцу
    df_2_grouped = df_2.groupby('accnt_id').sum()
    # Объединение df_1 и df_2_grouped по столбцу accnt_id
    df_merged = df_1.merge(df_2_grouped, on='accnt_id', how='left')


    # Класс для подготовки данных для PyTorch
    class CustomDataset(Dataset):
        def __init__(self, df, scaler):
            self.df = df
            self.scaler = scaler
            # Создание столбца age_group для группировки возраста
            self.df['age_group'] = (self.df['prsnt_age'] // 5) * 5
            # One-hot Encoding для категориальных признаков
            self.df = pd.get_dummies(self.df, columns=['gndr', 'accnt_status', 'addrss_type', 'rgn', 'phn',
                                                       'email', 'lk', 'assgn_npo', 'assgn_ops'],
                                     drop_first=True)
            # Замена булевых значений на 0 и 1
            self.df.replace({True: 1, False: 0}, inplace=True)
            # Удаление ненужных столбцов
            self.df.drop(columns=['slctn_nmbr', 'clnt_id', 'accnt_id', 'prsnt_age', 'accnt_bgn_date',
                                  'prvs_npf', 'brth_plc', 'dstrct',
                                  'city', 'sttlmnt', 'pstl_code', 'okato', 'oprtn_date'], inplace=True)
            # Масштабирование значений в столбцах sum, cprtn_prd_d, pnsn_age
            self.df[['sum', 'cprtn_prd_d', 'pnsn_age']] = self.scaler.fit_transform(
                self.df[['sum', 'cprtn_prd_d', 'pnsn_age']])

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            return torch.tensor(row.values, dtype=torch.float32)


    # Инициализация StandardScaler для масштабирования данных
    scaler = StandardScaler()
    # Создание объекта CustomDataset с использованием df_merged и scaler
    dataset = CustomDataset(df_merged, scaler)
    # Создание объекта DataLoader для загрузки данных по батчам
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

    batch_num = 0

    # Обработка данных по батчам
    for batch in tqdm(dataloader):
        # Преобразование батча в DataFrame
        batch_df = pd.DataFrame(batch.numpy(), columns=dataset.df.columns)
        # Сохранение обработанного батча в виде CSV-файла
        batch_file = os.path.join(output_dir, f"batch_{batch_num}.csv")
        batch_df.to_csv(batch_file, index=False)

        batch_num += 1
        # Очистка памяти
        del batch_df, batch
    all_batches = [pd.read_csv(os.path.join(output_dir, f)) for f in os.listdir(output_dir) if f.endswith(".csv")]
    df_merged_prepared_1 = pd.concat(all_batches, ignore_index=True)

    df_merged_prepared_1.to_csv("df_merged_prepared.csv", index=False)
    X = df_merged_prepared_1.drop(columns=['erly_pnsn_flg', 'cmmnt_Перевод в резерв Фонда (ОПС)',
                                           'cmmnt_Передача СПН в другой фонд по Уведомлениям ПФР о разделении ИЛС (ОПС)',
                                           'sum_type_РФОПС', 'rgn_БАЙКОНУР Г', 'rgn_БЕЛАРУСЬ', 'rgn_ЗАПОРОЖСКАЯ ОБЛ',
                                           'rgn_КАЗАХСТАН', 'rgn_РЕСП КАРЕЛИЯ', 'rgn_УКРАИНА', 'rgn_ЧУВАШСКАЯ РЕСП'])
    y = df_merged_prepared_1['erly_pnsn_flg']

    from_file = CatBoostClassifier()
    from_file.load_model("model")
    preds_class_val = from_file.predict(X)
    # Генерация и отображение roc-auc
    pred_df = pd.read_csv("pred_val.csv")
    metrics_df = metrics.metrics_df(y, preds_class_val)

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
