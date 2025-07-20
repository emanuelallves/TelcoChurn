# Análise de churn de clientes

Este projeto tem como objetivo analisar o comportamento dos clientes de uma empresa de telecomunicações e prever quais deles têm maior propensão a cancelar os serviços contratados (churn).

## Sobre o Projeto
- Objetivo: Desenvolver uma análise exploratória aprofundada e construir um modelo preditivo para identificar clientes com maior probabilidade de churn.
- Dataset: Telco Customer Churn.
- Técnicas utilizadas: análise exploratória (EDA), visualizações com gráficos, engenharia de atributos, treinamento de modelos de machine learning.

## Tecnologias e Bibliotecas Utilizadas
- Python (Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, imblearn)
- Git e GitHub

## Análise exploratória (EDA)
Durante a EDA, foram respondidas as seguintes perguntas de negócio:

1. Qual a proporção de clientes que deram churn?
Análise da taxa de churn em relação ao total de clientes.

2. Como é a distribuição dos gastos mensais?
Avaliação de simetria, concentração e presença de outliers nos gastos mensais dos clientes.

3. Como os métodos de pagamento estão distribuídos?
Visualização do total de clientes por método de pagamento.

4. Qual o perfil dos clientes?
Análise da distribuição das variáveis:

Gênero \n
Presença de parceiro\n
Senioridade (cliente idoso ou não)\n
Presença de dependentes\n

5. Como os gastos mensais variam entre clientes que deram churn e os que não deram?
Comparação da distribuição de gastos com foco em diferenças entre os grupos.

6. Qual a distribuição da variável tenure entre os grupos de churn?
Análise do tempo de permanência na empresa e sua relação com o churn.

7. Como questões familiares impactam o churn?
Avaliação da proporção de churn entre clientes com e sem parceiros/dependentes.

8. Existe interação entre gênero e presença de parceiro no churn?
Mapa de calor para analisar o churn em 4 combinações possíveis de gênero e presença de parceiro.

9. Quais são as maiores correlações entre variáveis numéricas?
Geração de heatmap de correlação e análise das 5 maiores correlações absolutas usando o coeficiente de Pearson.

## Modelagem preditiva
Para a etapa de modelagem, o foco foi construir um classificador que conseguisse prever com o máximo de assertividade os clientes com risco de churn. O processo foi iterativo e guiado por testes comparativos, com foco em métricas de classificação — principalmente recall, por ser mais crítico nesse problema (é mais perigoso não identificar um cliente que vai sair do que prever erroneamente que um cliente vai sair).

Testes Iniciais (Testes 1 a 4):
- Modelos testados: Árvore de Decisão e SVC.
- Codificações testadas: standardscaler, OrdinalEncoder e OneHotEncoder.
- Resultado: As métricas foram baixas, com F1-score em torno de 0.50~0.56. A mudança de codificador não teve impacto significativo. O SVC teve leve melhora na acurácia, mas ainda com recall abaixo de 0.50.

Introdução da random forest (Testes 5 a 7):
- Modelo: RandomForestClassifier com hiperparâmetros padrão.
- Resultado: Leve melhoria no F1-score (~0.53), mas ainda com recall em torno de 0.48.

Tunagem de Hiperparâmetros com GridSearchCV (Testes 8 a 11):
- Testei tunagem de parâmetros para Árvore de Decisão, SVC e Random Forest, ajustando também o parâmetro scoring para otimizar recall.
- Melhor resultado dessa fase: Árvore de Decisão tunada (Teste 8), com recall = 0.59 e f1-score = 0.61, mantendo uma boa acurácia (0.79).

Aplicando Oversampling com SMOTE (Testes 12 a 14):
- Para lidar com o desequilíbrio da variável alvo, utilizei o SMOTE para balancear a base.
- Os modelos foram treinados novamente com GridSearchCV e foco em recall.

Destaque final:
- A Random Forest com SMOTE (Teste 14) atingiu o melhor equilíbrio entre recall (0.73) e f1-score (0.62), com acurácia razoável (0.76).
- Esse modelo mostrou ser mais robusto para identificar clientes com maior risco de churn sem perder tanto em precisão.

Modelo Final Escolhido
- Modelo: RandomForestClassifier
- Pré-processamento: StandardScaler + OneHotEncoder
- Balanceamento: SMOTE aplicado na base de treino
- Tunagem: Hiperparâmetros otimizados com GridSearchCV (com scoring='recall')
- Métricas Finais:
  - Precision: 0.54
  - Recall: 0.73
  - F1-score: 0.62
  - Acurácia: 0.76

Matriz de Confusão do Modelo Final

A matriz de confusão abaixo mostra o desempenho do modelo final em classificar corretamente os clientes com e sem churn.

![Matriz de Confusão](CM.png)
