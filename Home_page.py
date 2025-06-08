import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from PIL import Image
from tensorflow.keras.models import load_model
from catboost import CatBoostClassifier

@st.cache_data
def load_data():
    return pd.read_csv("heart_main.csv")

@st.cache_resource
def load_models():
    models = {}
    model_files = {
        "KNN": "model_knn.pkl",
        "Gradient Boosting": "model_gb.pkl",
        "CatBoost": "model_catboost.pkl",
        "Random Forest": "model_rf.pkl",
        "Stacking": "model_stacking.pkl"
    }

    for name, file in model_files.items():
        try:
            with open(file, 'rb') as f:
                models[name] = pickle.load(f)
        except Exception as e:
            st.warning(f"Ошибка загрузки {name}: {e}")

    try:
        with open("nn_preprocessor.pkl", "rb") as f:
            nn_preprocessor = pickle.load(f)
        nn_model = load_model("model_nn.h5", compile=False)
        models["Neural Network"] = (nn_preprocessor, nn_model)
    except Exception as e:
        st.warning(f"Ошибка загрузки нейросети: {e}")

    return models

def info_page():
    st.title('Информация')

    st.markdown("ФИО: Колодницкий Илья Михайлович")
    st.markdown("Группа: ФИТ-232")
                
    st.image("therock.jpg", width=900, caption="Я на пляже")

    st.subheader("Тема РГР:  :blue[«Разработка Web-приложения (дашборда) для инференса (вывода) моделей ML и анализа данных»] :sunglasses:")

def data_page():
    st.title("Информация о наборе данных")
    data = load_data()

    st.header("Описание предметной области")
    st.write("""
    Цель данного проекта — предсказать наличие сердечного заболевания на основе медицинских показателей. 
    Сердечно-сосудистые заболевания являются одной из ведущих причин смертности во всём мире. 
    Этот датасет собран из клинических данных пациентов и используется для построения моделей машинного обучения, 
    способных выявить возможное заболевание на ранних этапах.
    """)

    st.header("Описание признаков")
    st.write("""
    - **age**: возраст пациента  
    - **sex**: пол  
    - **cp**: тип боли в груди  
    - **trestbps**: артериальное давление в покое  
    - **chol**: уровень холестерина  
    - **fbs**: уровень сахара натощак  
    - **restecg**: результаты ЭКГ  
    - **thalach**: максимальный пульс  
    - **exang**: стенокардия при физической нагрузке  
    - **oldpeak**: депрессия ST  
    - **slope**: наклон сегмента ST  
    - **ca**: количество крупных сосудов  
    - **thal**: результат теста таллия  
    - **target (или num)**: наличие сердечного заболевания
    """)


    st.header("Особенности предобработки данных")
    st.write("""
    - Удаление/замена пропущенных значений  
    - Нормализация числовых признаков с помощью StandardScaler  
    - Кодирование категориальных признаков с помощью OneHotEncoder/LabelEncoder  
    - Балансировка классов (например, через SMOTE)  
    - Разделение на обучающую и тестовую выборки  
    """)

    st.header("Разведочный анализ данных (EDA)")
    st.subheader("Распределение целевой переменной")
    fig1, ax1 = plt.subplots()
    sns.countplot(x='num', data=data, ax=ax1)
    ax1.set_title("Наличие сердечного заболевания")
    st.pyplot(fig1)

    st.subheader("Распределение возраста")
    fig2, ax2 = plt.subplots()
    sns.histplot(data['age'], bins=20, kde=True, ax=ax2)
    ax2.set_title("Возраст пациентов")
    st.pyplot(fig2)

    st.subheader("Максимальный пульс vs Возраст")
    fig3, ax3 = plt.subplots()
    sns.scatterplot(x='age', y='thalach', hue='num', data=data, ax=ax3)
    ax3.set_title("Связь между возрастом и пульсом")
    st.pyplot(fig3)

    st.subheader("Корреляционная матрица признаков")
    fig4, ax4 = plt.subplots(figsize=(10, 7))
    sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax4)
    st.pyplot(fig4)

    st.header("Пример данных")
    st.dataframe(data.head())


def graphics_page():
    st.title("Визуализация данных")
    data = load_data()

    st.header("1. Распределение целевой переменной")
    fig, ax = plt.subplots()
    sns.countplot(x='num', data=data, ax=ax)
    ax.set_title("Распределение по наличию сердечного заболевания")
    st.pyplot(fig)

    st.header("2. Корреляционная матрица")
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.header("3. Распределение возраста")
    fig, ax = plt.subplots()
    sns.histplot(data['age'], bins=20, kde=True, ax=ax)
    st.pyplot(fig)

    st.header("4. Возраст vs Максимальный пульс")
    fig, ax = plt.subplots()
    sns.scatterplot(x='age', y='thalach', hue='num', data=data, ax=ax)
    st.pyplot(fig)

    st.header("5. Boxplot уровня холестерина")
    fig, ax = plt.subplots()
    sns.boxplot(x='num', y='chol', data=data, ax=ax)
    st.pyplot(fig)


def predict_page():
    st.title("Предсказание диагноза")
    models = load_models()
    if not models:
        st.error("Модели не загружены")
        return

    model_choice = st.selectbox("Выберите модель", list(models.keys()))

    st.subheader("1. Загрузка файла с данными")
    uploaded_file = st.file_uploader("Загрузите CSV-файл для пакетного предсказания", type=["csv"])
    if uploaded_file:
        try:
            input_df = pd.read_csv(uploaded_file)

            if model_choice == "CatBoost":
                for col in ['cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']:
                    if col in input_df.columns:
                        input_df[col] = input_df[col].astype(str)

            if st.button("Предсказать из файла"):
                if model_choice == "Neural Network":
                    preprocessor, nn_model = models[model_choice]
                    input_transformed = preprocessor.transform(input_df)
                    prediction = (nn_model.predict(input_transformed) > 0.5).astype(int)
                    input_df["Предсказание"] = prediction
                else:
                    model = models[model_choice]
                    prediction = model.predict(input_df)
                    input_df["Предсказание"] = prediction

                st.subheader("Результаты предсказания")
                st.dataframe(input_df)
                return  

        except Exception as e:
            st.error(f"Ошибка при обработке файла: {e}")

    st.subheader("2. Ручной ввод данных")

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Возраст", 1, 120, 50)
        sex = st.radio("Пол", ["Мужской", "Женский"])
        cp = st.selectbox("Тип боли", ["Типичная", "Атипичная", "Неангинозная", "Бессимптомно"])
        trestbps = st.number_input("Давление (мм рт.ст.)", 80, 200, 120)
        chol = st.number_input("Холестерин (мг/дл)", 100, 600, 200)
        fbs = st.radio("Сахар > 120", ["Да", "Нет"])
    
    with col2:
        restecg = st.selectbox("ЭКГ", ["Норма", "ST-T", "Гипертрофия"])
        thalach = st.number_input("Макс. пульс", 60, 220, 150)
        exang = st.radio("Стенокардия при нагрузке", ["Да", "Нет"])
        oldpeak = st.number_input("Oldpeak", 0.0, 10.0, 1.0, 0.1)
        slope = st.selectbox("Наклон ST", ["Восходящий", "Плоский", "Нисходящий"])
        ca = st.number_input("Сосуды", 0, 3, 0)
        thal = st.selectbox("Таллий", ["Норма", "Фиксированный", "Обратимый"])

    sex = 1 if sex == "Мужской" else 0
    cp = {"Типичная": 1, "Атипичная": 2, "Неангинозная": 3, "Бессимптомно": 4}[cp]
    fbs = 1 if fbs == "Да" else 0
    restecg = {"Норма": 0, "ST-T": 1, "Гипертрофия": 2}[restecg]
    exang = 1 if exang == "Да" else 0
    slope = {"Восходящий": 1, "Плоский": 2, "Нисходящий": 3}[slope]
    thal = {"Норма": 3, "Фиксированный": 6, "Обратимый": 7}[thal]

    input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg,
                            thalach, exang, oldpeak, slope, ca, thal]],
                          columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
                                   'restecg', 'thalach', 'exang', 'oldpeak',
                                   'slope', 'ca', 'thal'])

    if st.button("Предсказать"):
        if model_choice == "Neural Network":
            try:
                preprocessor, nn_model = models[model_choice]
                input_transformed = preprocessor.transform(input_data)
                prediction = (nn_model.predict(input_transformed) > 0.5).astype(int)
                proba = [1 - prediction[0][0], prediction[0][0]]
            except Exception as e:
                st.error(f"Ошибка при обработке нейросети: {e}")
                return
        elif model_choice == "CatBoost":
            model = models[model_choice]
            input_df = pd.DataFrame(input_data, columns=[
                'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
            ])
            for col in ['cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']:
                input_df[col] = input_df[col].astype(str)

            prediction = model.predict(input_df)
            proba = model.predict_proba(input_df)[0]
        else:
            model = models[model_choice]
            prediction = model.predict(input_data)
            proba = model.predict_proba(input_data)[0]

        st.subheader("Результат")
        if prediction[0] == 1:
            st.error(f"Риск заболевания: {proba[1] * 100:.2f}%")
        else:
            st.success(f"Вероятность отсутствия болезни: {proba[0] * 100:.2f}%")

def main():
    st.set_page_config(page_title="Анализ сердца", layout="wide")
    st.sidebar.title("Меню навигации")
    page = st.sidebar.radio("Выберите страницу", 
                            ["Информация", "Данные", "Визуализация", "Предсказание"])
    
    if page == "Информация":
        info_page()
    elif page == "Данные":
        data_page()
    elif page == "Визуализация":
        graphics_page()
    elif page == "Предсказание":
        predict_page()

if __name__ == "__main__":
    main()
