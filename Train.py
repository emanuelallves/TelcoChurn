# %%
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, make_scorer, recall_score
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import mlflow

# %%
mlflow.set_tracking_uri('http://127.0.0.1:5000/')
mlflow.set_experiment(experiment_id='832769808834908456')

# %%
DATASET_PATH = '../Data/WA_Fn-UseC_-Telco-Customer-Churn-clean.csv'

# %%
df = pd.read_csv(DATASET_PATH)
df.info()

# %%
features = df.drop(columns='Churn')
target = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(features,
                                                    target,
                                                    test_size=0.2,
                                                    random_state=42,
                                                    stratify=target)

X_train, X_val, y_train, y_val = train_test_split(X_train,
                                                  y_train,
                                                  test_size=0.25,
                                                  random_state=42,
                                                  stratify=y_train)

print(f'X_train shape: {X_train.shape}\ny_train shape: {y_train.shape}\n')
print(f'X_test shape: {X_test.shape}\ny_test shape: {y_test.shape}\n')
print(f'X_val shape: {X_val.shape}\ny_val shape: {y_val.shape}\n')

df_test = X_test.copy()
df_test['Churn'] = y_test

df_test.to_csv('test_set.csv', index=False)

# %%
numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
categorical_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents',
        'PhoneService', 'MultipleLines', 'InternetService',
       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
       'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
       'PaymentMethod']

#%%
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(), categorical_cols)
    ]
)

pipe_tree = Pipeline([
    ('preprocessing', preprocessor),
    ('model', DecisionTreeClassifier(random_state=42))
])

pipe_rf = Pipeline([
    ('preprocessing', preprocessor),
    ('model', RandomForestClassifier(random_state=42))
])

pipe_svc = Pipeline([
    ('preprocessing', preprocessor),
    ('model', SVC(random_state=42))
])

# %%
smote = SMOTE(random_state=42)

imb_pipe_tree = ImbPipeline([
    ('preprocessing', preprocessor),
    ('smote', smote),
    ('model', DecisionTreeClassifier(random_state=42))
])

imb_pipe_rf = ImbPipeline([
    ('preprocessing', preprocessor),
    ('smote', smote),
    ('model', RandomForestClassifier(random_state=42))
])

imb_pipe_svc = ImbPipeline([
    ('preprocessing', preprocessor),
    ('smote', smote),
    ('model', SVC(random_state=42))
])

# %%
param_grid_tree = {
    'model__criterion': ['gini', 'entropy', 'log_loss'],
    'model__max_depth': [None, 5, 10, 20],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 4],
    'model__max_features': [None, 'sqrt', 'log2']
}

param_grid_svc = {
    'model__C': [0.1, 1, 10],
    'model__kernel': ['linear', 'rbf', 'poly'],
    'model__gamma': ['scale', 'auto']
}

param_grid_rf = {
    'model__n_estimators': [350, 400, 450],
    'model__max_depth': [7, 9, 10, 13],
    'model__min_samples_split': [2, 3, 5, 7],
    'model__min_samples_leaf': [1, 2, 3],
    'model__max_features': ['sqrt', 'log2']
}

recall_yes = make_scorer(recall_score, pos_label='Yes')

grid_search = GridSearchCV(imb_pipe_rf,
                           param_grid_rf,
                           cv=5,
                           scoring=recall_yes,
                           n_jobs=-1)

# %%
with mlflow.start_run():
    mlflow.sklearn.autolog()

    grid_search.fit(X_train, y_train)

    y_pred_val = grid_search.predict(X_val)

    report = classification_report(y_val, y_pred_val, output_dict=True)

    precision_yes = report['Yes']['precision']
    recall_yes = report['Yes']['recall']
    f1_yes = report['Yes']['f1-score']

    mlflow.log_metrics({'precision_yes': precision_yes,
                        'recall_yes': recall_yes,
                        'f1_yes': f1_yes})

# %%
cm = confusion_matrix(y_val, y_pred_val, labels=['No', 'Yes'])

sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=['No', 'Yes'],
            yticklabels=['No', 'Yes'],
            cmap='Blues')
plt.xlabel('Predito')
plt.ylabel('Real')
plt.title('Matriz de Confusão')
plt.show()

# %%
grid_search.best_params_

# %%
y_pred_test = grid_search.predict(X_test)

print(classification_report(y_test, y_pred_test))

cm = confusion_matrix(y_test, y_pred_test, labels=['No', 'Yes'])

sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=['No', 'Yes'],
            yticklabels=['No', 'Yes'],
            cmap='Blues')
plt.xlabel('Predito')
plt.ylabel('Real')
plt.title('Matriz de Confusão')
plt.savefig("CM.png", dpi=300, bbox_inches='tight')
plt.show()
# %%
pd.set_option('display.max_columns', None)
df.head()
# %%
df.isna().sum()
# %%
