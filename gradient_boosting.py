import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

# Pasta onde os arquivos CSV estão localizados
PASTA_DATASET = "dataset min. turismo"

# Lista para armazenar os DataFrames de cada arquivo CSV
dfs = []

# Loop através dos anos disponíveis
anos_disponíveis = list(range(1989, 2023))  # Ajuste o intervalo de anos conforme necessário
for ano in anos_disponíveis:
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

# Combine todos os DataFrames em um único DataFrame
dataframe_completo = pd.concat(dfs, ignore_index=True)

dataframe_completo['Chegadas'].fillna(0, inplace=True)

features = ['Ordem continente', 'Ordem país', 'Ordem UF', 'Ordem mês', 'Ordem via de acesso', 'ano']
target = 'Chegadas'

X = dataframe_completo[features]
y = dataframe_completo[target]

# Codificar variáveis categóricas (strings) em números
X = pd.get_dummies(X, drop_first=True)

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar o modelo de Gradient Boosting Regressor
model = GradientBoostingRegressor(random_state=42)

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar o modelo com os dados de treinamento
model.fit(X_train, y_train)

# Fazer previsões
y_pred = model.predict(X_test)

# Avaliar o modelo
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f"RMSE sem ajuste de hiperparâmetros: {rmse}")
print(f"R² sem ajuste de hiperparâmetros: {r2}")

# Definir os hiperparâmetros a serem ajustados
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5]
}

# Realizar uma busca em grade para encontrar os melhores hiperparâmetros
grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Imprimir os melhores hiperparâmetros
print("Melhores hiperparâmetros:", grid_search.best_params_)

# Criar um modelo com os melhores hiperparâmetros
best_model = grid_search.best_estimator_

# Treinar o modelo com os dados de treinamento
best_model.fit(X_train, y_train)

# Fazer previsões
y_pred = best_model.predict(X_test)

# Avaliar o modelo
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f"RMSE com os melhores hiperparâmetros: {rmse}")
print(f"R² com os melhores hiperparâmetros: {r2}")
