# %%
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns
sns.set_palette("pastel")
import numpy as np

# %%
DATASET_PATH = '../Data/WA_Fn-UseC_-Telco-Customer-Churn.csv'

# %%
df = pd.read_csv(DATASET_PATH)
print(df.head())

# %%
print(f'O dataset possui {df.shape[0]} linhas e {df.shape[1]} colunas')
print(f'Possui {df.isna().any(axis=1).sum()} linhas com valores ausentes')
print(f'Linhas duplicadas: {df.duplicated().sum()}')
print(f'Distribuição da variável alvo:\n{df.Churn.value_counts(normalize=True)}')

# %%
plt.figure(figsize=(10, 6))
sns.countplot(x=df['Churn'])
plt.title('Distribuição da retenção de clientes')
plt.xlabel('Classes')
plt.ylabel('Contagem')
plt.show()

# %%
plt.figure(figsize=(10, 6))
sns.histplot(df['MonthlyCharges'], kde=True, bins=30)
plt.title('Distribuição dos Gastos Mensais')
plt.xlabel('Gasto Mensal (MonthlyCharges)')
plt.ylabel('Frequência')
plt.show()

# %%
plt.figure(figsize=(10, 6))
sns.countplot(data=df, y='PaymentMethod', order=df['PaymentMethod'].value_counts().index)
plt.title('Clientes por Método de Pagamento')
plt.xlabel('Número de Clientes')
plt.ylabel('Método de Pagamento')
plt.tight_layout()
plt.show()

# %%
cols = ['gender', 'Partner', 'SeniorCitizen', 'Dependents']
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

for col, ax in zip(cols, axs.ravel()):
    sns.countplot(data=df, x=col, ax=ax)
    ax.set_title(f'Distribuição de {col}')
    ax.set_xlabel(col)
    ax.set_ylabel('Contagem')

plt.tight_layout()
plt.show()

# %%
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Churn', y='MonthlyCharges')
plt.title('Gastos Mensais por Situação de Churn')
plt.xlabel('Churn')
plt.ylabel('Gastos Mensais')
plt.show()

# %%
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='tenure', hue='Churn', kde=True, bins=30, multiple="stack")
plt.title('Distribuição de Tenure por Churn')
plt.xlabel('Tenure (meses)')
plt.ylabel('Contagem')
plt.show()

# %%
dep_churn = pd.crosstab(df['Dependents'], df['Churn'], normalize='index') * 100
print(dep_churn)

partner_churn = pd.crosstab(df['Partner'], df['Churn'], normalize='index') * 100
print(partner_churn)

fig, axs = plt.subplots(1, 2, figsize=(14, 5))

sns.barplot(x='Dependents', hue='Churn', data=df, ax=axs[0])
axs[0].set_title('Churn por Dependentes')

sns.barplot(x='Partner', hue='Churn', data=df, ax=axs[1])
axs[1].set_title('Churn por Parceiro')

plt.tight_layout()
plt.show()

# %%
plt.figure(figsize=(10, 6))

heat_data = df.groupby(['Partner', 'gender'])['Churn'].value_counts(normalize=True).unstack().fillna(0)['Yes'] * 100

heat_df = heat_data.unstack().T

sns.heatmap(heat_df, annot=True, cmap='Blues', fmt=".1f")
plt.title('Percentual de Churn por Gênero e Parceiro')
plt.ylabel('Gênero')
plt.xlabel('Possui Parceiro')
plt.show()

# %%
num_cols = df.select_dtypes(include='number')

correlation_matrix = num_cols.corr(method='pearson')

sns.heatmap(correlation_matrix, annot=True, cmap='Blues', fmt=".2f", mask=np.triu(correlation_matrix))
plt.title('Mapa de Calor - Correlação entre Variáveis Numéricas')
plt.show()

corr_pairs = correlation_matrix.where(~np.eye(correlation_matrix.shape[0], dtype=bool)).stack().abs()
top_5_corr = corr_pairs.sort_values(ascending=False).head(5)
print("Top 5 correlações (em módulo):")
print(top_5_corr)

# %%
