import streamlit as st
import pandas as pd
import requests
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Classificador de Churn em lote")

uploaded_file = st.file_uploader("Carregue um arquivo CSV com os dados dos clientes", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Dados carregados:")
    st.dataframe(df)

    if st.button("Fazer predição"):
        try:
            required_columns = [
                "tenure", "MonthlyCharges", "TotalCharges",
                "gender", "SeniorCitizen", "Partner", "Dependents",
                "PhoneService", "MultipleLines", "InternetService",
                "OnlineSecurity", "OnlineBackup", "DeviceProtection",
                "TechSupport", "StreamingTV", "StreamingMovies",
                "Contract", "PaperlessBilling", "PaymentMethod"
            ]

            filtered_df = df[required_columns]

            data = filtered_df.to_dict(orient="records")
            response = requests.post("http://127.0.0.1:8000/predict", json=data)
            result = response.json()

            if "predictions" in result:
                df["Prediction"] = result["predictions"]
                st.success("Predições feitas com sucesso!")
                st.write(df['Prediction'].head())

                y_true = df["Churn"]
                y_pred = df["Prediction"]

                cm = confusion_matrix(y_true, y_pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                            xticklabels=["No", "Yes"],
                            yticklabels=["No", "Yes"],
                            ax=ax)
                plt.xlabel('Predito')
                plt.ylabel('Real')
                plt.title('Matriz de Confusão')

                st.pyplot(fig)

                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Baixar CSV com predições",
                    data=csv,
                    file_name="predicoes_churn.csv",
                    mime='text/csv',
                )
            else:
                st.error("Erro ao obter predições.")

        except Exception as e:
            st.error("Erro ao processar as predições.")
            st.error(str(e))
