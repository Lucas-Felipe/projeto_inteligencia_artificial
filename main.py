"""Arquivo main"""
import os
import pandas as pd

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
print("DataFrame completo:")
print(dataframe_completo)
