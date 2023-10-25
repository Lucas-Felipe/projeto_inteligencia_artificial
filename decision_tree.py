"""Arquivo main"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Pasta onde os arquivos CSV estão localizados
PASTA_DATASET = "dataset min. turismo"

# Lista para armazenar os DataFrames de cada arquivo CSV
dfs = []

# Loop através dos anos disponíveis
anos_disponiveis = list(range(1989, 2023))  # Ajuste o intervalo de anos conforme necessário
for ano in anos_disponiveis:

    arquivo_csv = os.path.join(PASTA_DATASET, f"chegadas_{ano}.csv")

    # Verifica se o arquivo existe antes de tentar lê-lo
    try:
        # Ler o arquivo com a codificação 'utf-8' e delimitador ';'
        df = pd.read_csv(arquivo_csv, encoding='utf-8', delimiter=';', header=0,
                         converters={'País': lambda x: x.replace(', ', '|')})
        dfs.append(df)

    except FileNotFoundError:
        print(f"Arquivo {arquivo_csv} não encontrado.")
    except UnicodeDecodeError:
        print(f"Erro de codificação ao ler o arquivo {arquivo_csv}.")

for i, df in enumerate(dfs):
    dfs[i] = dfs[i].apply(lambda x: x.str.lower() 
                          if x.name in dfs[i].select_dtypes(include='object') else x)

for df in dfs:
    df.rename(columns={
        'Ano': 'ano',
        'cod mes': 'Ordem mês',
        'cod continente': 'Ordem continente',
        'cod pais': 'Ordem país',
        'cod uf': 'Ordem UF',
        'Via': 'Via de acesso',
        'cod via': 'Ordem via de acesso'
    }, inplace=True)

# for i in dfs:
#     print(i)
# Combine todos os DataFrames em um único DataFrame
dataframe_completo = pd.concat(dfs, ignore_index=True)

# Agora você tem um único DataFrame contendo todos os dados, exceto o ano de 2016
# print("DataFrame completo:")
# print(dataframe_completo)

dataframe_completo['Chegadas'].fillna(0, inplace=True)

features = ['Ordem continente', 'Ordem país', 'Ordem UF', 'Ordem mês', 'Ordem via de acesso', 'ano']
target = 'Chegadas'

X = dataframe_completo[features]
y = dataframe_completo[target]

# Codificar variáveis categóricas (strings) em números
X = pd.get_dummies(X, drop_first=True)

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar o modelo de árvore de decisão
model = DecisionTreeRegressor()

# Treinar o modelo
model.fit(X_train, y_train)

# Fazer previsões
y_pred = model.predict(X_test)

# Avaliar o modelo
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse}")
print(f"R²: {r2}")

# Defina a grade de hiperparâmetros que deseja testar
param_grid = {
    'max_depth': [None, 10, 20, 30],  # Profundidade máxima da árvore
    'min_samples_split': [2, 5, 10],  # Número mínimo de amostras necessárias para dividir um nó
    'min_samples_leaf': [1, 2, 4],   # Número mínimo de amostras necessárias em um nó folha
    'max_features': [None, 'sqrt', 'log2']  # Número máximo de recursos a serem considerados para a divisão
}

# Crie o objeto GridSearchCV
grid_search = GridSearchCV(DecisionTreeRegressor(), param_grid, cv=5, scoring='neg_mean_squared_error')

# Realize a busca em grade no conjunto de treinamento
grid_search.fit(X_train, y_train)

# Obtenha os melhores hiperparâmetros encontrados
best_params = grid_search.best_params_
print("Melhores hiperparâmetros:", best_params)

# Crie um novo modelo com os melhores hiperparâmetros
best_model = DecisionTreeRegressor(**best_params)

# Treine o modelo com os melhores hiperparâmetros
best_model.fit(X_train, y_train)

# Faça previsões com o modelo ajustado
y_pred_best = best_model.predict(X_test)

# Avalie o modelo com os melhores hiperparâmetros
rmse_best = mean_squared_error(y_test, y_pred_best, squared=False)
r2_best = r2_score(y_test, y_pred_best)

print("RMSE com os melhores hiperparâmetros:", rmse_best)
print("R² com os melhores hiperparâmetros:", r2_best)
