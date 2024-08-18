import os
import pandas as pd

# Funzione per descrivere il dataframe
def describe(df):
    variables = []
    dtypes = []
    count = []
    unique = []
    missing = []

    for item in df.columns:
        variables.append(item)
        dtypes.append(df[item].dtype)
        count.append(len(df[item]))
        unique.append(len(df[item].unique()))
        missing.append(df[item].isna().sum())

    output = pd.DataFrame({
        'variabile': variables,
        'dtype': dtypes,
        'conteggio': count,
        'unico': unique,
        'valore mancante': missing
    })

    return output

# Funzione per rimuovere entità dal dataframe
def remove_entities(df, entities_to_remove):
    return df[~df['Entity'].isin(entities_to_remove)]

# Directory corrente
current_dir = os.path.dirname(os.path.abspath(__file__))

# Definisci i percorsi dei file CSV
file1 = os.path.join(current_dir, '..', 'DbOriginali', 'GlobalLandTemperaturesByCity.csv')
file2 = os.path.join(current_dir, '..', 'DbOriginali', 'GlobalLandTemperaturesByCountry.csv')
file4 = os.path.join(current_dir, '..', 'DbOriginali', 'GlobalLandTemperaturesByMajorCity.csv')
file5 = os.path.join(current_dir, '..', 'DbOriginali', 'GlobalLandTemperaturesByState.csv')
file7 = os.path.join(current_dir, '..', 'DbOriginali', 'GlobalTemperatures.csv')
filesec = os.path.join(current_dir, '..', 'DbOriginali', 'SocialEconomicData.csv')

# Dizionario dei percorsi dei file
file_paths = {
    'file1': file1,
    'file2': file2,
    'file4': file4,
    'file5': file5,
    'file7': file7,
    'filesec': filesec
}

# Verifica se tutti i file esistono
if all(os.path.exists(path) for path in file_paths.values()):

    # Leggi i file CSV
    df1 = pd.read_csv(file_paths['file1'])
    df2 = pd.read_csv(file_paths['file2'])
    df4 = pd.read_csv(file_paths['file4'])
    df5 = pd.read_csv(file_paths['file5'])
    df7 = pd.read_csv(file_paths['file7'])
    df_sec = pd.read_csv(file_paths['filesec'])

    # Unisci df1 e df2
    df_merge = pd.merge(df1, df2, on=['Entity', 'Code', 'Year'], how='inner')

    # Entità da rimuovere dall'unione
    entities_to_remove_merge = ['World', 'Africa (IHME GBD)', 'America (IHME GBD)', 'Asia (IHME GBD)', 'Europe (IHME GBD)', 'European Union (27)', 'Australia', 'Low-income countries', 'High-income countries', 'Lower-middle-income countries']
    df_merge = remove_entities(df_merge, entities_to_remove_merge)

    # Percorso del file di output per i dati uniti
    output_file_path_merge = os.path.join(current_dir, '..', 'DbDefinitivi', 'climateChange.csv')
    df_merge.to_csv(output_file_path_merge, index=False)
    print(f"Il file è stato salvato con successo in: {output_file_path_merge}")

    #4,5,7
    # Rimuovi entità da df4 e df5
    entities_to_remove_4 = ['World']
    entities_to_remove_5 = ['Beijing/Shanghai, China', 'High-income countries', 'Lower-middle-income countries', 'Medellin, Colombia', 'Murcia, Spain', 'Sao Paulo, Brazil', 'Upper-middle-income countries']
    df4 = remove_entities(df4, entities_to_remove_4)
    df5 = remove_entities(df5, entities_to_remove_5)

    # Elimina la colonna 'Code' da df4 e df7
    df4.drop(columns=['Code'], inplace=True)
    df7.drop(columns=['Code'], inplace=True)

    # Dati socio-economici
    df_sec.rename(columns={
        #'Country Name': 'Entity',
        'Country Code': 'Code'
    }, inplace=True)

    df_sec.drop(columns=['2022 [YR2022]','2023 [YR2023]'], inplace=True)

    output_file_path_dfe = os.path.join(current_dir, '..', 'DbDefinitivi', 'SocialEconomicData.csv')
    df_sec.to_csv(output_file_path_dfe, index=False)
    print(f"Il file è stato salvato con successo in: {output_file_path_dfe}")

else:
    print(f"Uno dei file non esiste.")
