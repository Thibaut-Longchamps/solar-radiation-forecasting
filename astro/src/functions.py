import os
import pandas as pd
import numpy as np
import boto3
import sys
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
import joblib
# import boto3
from datetime import datetime
import requests
from io import StringIO
import warnings
from io import StringIO
import os
from dotenv import load_dotenv
import logging
from itertools import combinations




# Charger les variables d'environnement à partir du fichier .env
load_dotenv()

# Accéder aux variables
aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')

from airflow.providers.amazon.aws.hooks.s3 import S3Hook

def download_file_from_s3(bucket_name: str, s3_key: str, local_path: str, aws_conn_id: str = 'aws_s3_conn'):
    """
    Télécharge un fichier spécifique depuis un bucket S3 et l'enregistre dans un emplacement local.

    Parameters:
        bucket_name (str): Le nom du bucket S3.
        s3_key (str): Le chemin du fichier dans S3 (clé S3).
        local_path (str): Le chemin local où le fichier doit être enregistré.
        aws_conn_id (str): L'identifiant de la connexion AWS dans Airflow.
    """
    hook = S3Hook(aws_conn_id=aws_conn_id)
    hook.get_key(s3_key, bucket_name).download_file(local_path)
    print(f"Fichier {s3_key} téléchargé depuis le bucket {bucket_name} vers {local_path}")

def copy_file_to_volume(source_path: str, destination_path: str):
    """
    Copie un fichier local vers un chemin spécifique dans le conteneur.

    Parameters:
        source_path (str): Le chemin du fichier source sur la machine hôte.
        destination_path (str): Le chemin de destination dans le volume monté du conteneur.
    """
    shutil.copy(source_path, destination_path)
    print(f"Fichier copié de {source_path} vers {destination_path}")   



def save_data_to_volume(df: pd.DataFrame, file_path: str):
    """
    Enregistre le DataFrame transformé dans un volume Docker sous forme de fichier CSV.

    Parameters:
        df (pd.DataFrame): Le DataFrame contenant les données transformées.
        file_path (str): Le chemin local dans le volume Docker où le fichier sera enregistré.
    """
    df.to_csv(file_path, index=False)
    print(f"Données transformées enregistrées dans {file_path}")
    


def import_and_transform_csv(file_path):
    # Charger le fichier CSV depuis le volume monté
    df = pd.read_csv(file_path)
    # Transformation de base, exemple : renommer les colonnes en minuscules
    df.columns = [col.lower() for col in df.columns]
    return df



def import_and_transform_csv(file_path):
    # Charger le fichier CSV depuis le volume monté
    df = pd.read_csv(file_path)
    return df

def download_from_s3(bucket_name, s3_key, local_dir, aws_access_key, aws_secret_key):
    """
    Télécharge un fichier depuis un bucket S3 vers un dossier local spécifié et le charge dans un DataFrame.

    Args:
        bucket_name (str): Le nom du bucket S3.
        s3_key (str): Le chemin du fichier dans S3.
        local_dir (str): Le dossier local pour sauvegarder le fichier.
        aws_access_key (str): Clé d'accès AWS.
        aws_secret_key (str): Clé secrète AWS.

    Returns:
        pd.DataFrame: Le fichier téléchargé chargé dans un DataFrame.
    """
    session = boto3.Session(
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key
    )
    s3 = session.resource('s3')

    # Extraire le nom du fichier depuis le chemin s3_key
    filename = os.path.basename(s3_key)
    local_path = os.path.join(local_dir, filename)
    
    try:
        # Téléchargement du fichier depuis S3
        s3.Bucket(bucket_name).download_file(s3_key, local_path)
        print(f"Fichier {s3_key} téléchargé avec succès depuis S3 vers {local_path}.")
        
        # Charger le fichier dans un DataFrame
        df = pd.read_csv(local_path)
        print(f"Le fichier {filename} a été chargé dans un DataFrame avec succès.")
        return df

    except Exception as e:
        print(f"Erreur lors du téléchargement ou du chargement de {s3_key} depuis S3 : {e}")
        raise

    


def keep_useful_columns(data, columns):
    """
    Garde les colonnes utiles dans un DataFrame.

    :param data: DataFrame pandas dans lequel certaines colonnes seront conservées.
    :param columns: Liste des noms des colonnes à conserver.
    :return: DataFrame pandas contenant uniquement les colonnes spécifiées, ou None en cas d'erreur.
    """
    try:
        # Sélectionner les colonnes utiles
        data = data[columns]
        return data  # Retourner le DataFrame si tout va bien
    except KeyError as e:
        # Gérer l'erreur et retourner None
        print(f"Erreur : Les colonnes suivantes sont manquantes : {e}")
        return None
    


def merge_dataframes(df1, df2, key, how='left'):
    """
    Fusionne deux DataFrames sur une clé spécifique.

    :param df1: Premier DataFrame à fusionner.
    :param df2: Second DataFrame à fusionner.
    :param key: Colonne(s) clé(s) pour la fusion (peut être une chaîne de caractères ou une liste).
    :param how: Type de fusion (left, right, inner, outer). Par défaut, 'left'.
    :param suffixes: Tuple de suffixes à utiliser pour les colonnes ayant le même nom. Par défaut, ('_x', '_y').
    :return: DataFrame fusionné.
    """
    # Fusionner les DataFrames
    merged_df = pd.merge(df1, df2, on=key, how=how)
    
    return merged_df



def process_merged_dataframe(df):
    """
    Traite le DataFrame fusionné en appliquant les transformations suivantes :
    1. Convertir la colonne 'date' en type datetime.
    2. Renommer la colonne 'kWh' en 'GHI'.
    3. Extraire les caractéristiques temporelles (jour, mois, année, heure, jour de la semaine).
    4. Trier le DataFrame par 'name' et 'date'.
    
    :param df: DataFrame à traiter.
    :return: DataFrame traité avec les nouvelles colonnes temporelles.
    """
    # Convertir la colonne 'date' en datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Renommer la colonne 'kWh' en 'GHI'
    df.rename(columns={'kWh': 'GHI'}, inplace=True)
    
    # Extraire les caractéristiques temporelles
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['hour'] = df['date'].dt.hour
    df['weekday'] = df['date'].dt.weekday  # 0=Monday, 6=Sunday
    
    # Trier par 'name' et 'date'
    df = df.sort_values(by=['name', 'date'])
    
    return df



def convert_datetime(data, column):
    """
    Converti la colonne spécifié au format datetime
    
    :param data: dataFrame à traiter
    :param column: colonne à convertir
    :return: DataFrame modifié
    """
    try:
        data[column] = pd.to_datetime(data[column])
    except Exception as e:
        print(f"Erreur lors de la conversion de la colonne {column}: {e}")
    return data





def rename_and_sort(data, old_col:str, new_col:str, sort_by_columns:list):
    """
    Renomme une colonne d'un DataFrame et le trie selon des colonnes spécifiques.
    
    :param data: DataFrame à traiter.
    :param old_col: Nom de la colonne à renommer.
    :param new_col: Nouveau nom de la colonne.
    :param sort_by_columns: Liste des colonnes pour trier le DataFrame.
    :return: DataFrame modifié.
    """
    # Renommer la colonne
    data.rename(columns={old_col: new_col}, inplace=True)
    
    # Trier le DataFrame
    data = data.sort_values(by=sort_by_columns)
    return data



def extract_temporality(data, date_column:str):
    """
    Extrait les caractéristiques temporelles d'une colonne date et les ajoute au DataFrame.
    
    :param data: DataFrame contenant la colonne date
    :param date_column: Nom de la colonne contenant les dates
    :return: DataFrame modifié avec les nouvelles colonnes temporelles
    """
    # Assurez-vous que la colonne est bien au format datetime
    data[date_column] = pd.to_datetime(data[date_column])

    # Création de nouvelles caractéristiques temporelles
    data['year'] = data[date_column].dt.year
    data['month'] = data[date_column].dt.month
    data['day'] = data[date_column].dt.day
    data['hour'] = data[date_column].dt.hour
    data['weekday'] = data[date_column].dt.weekday
    
    return data

def api_new(start_date, end_date, latitude, longitude):
    date_string_1 = start_date + "T00:00"
    date_string_2 = end_date + "T00:00"
    x = [(latitude, longitude)]
    li = []
    
    for i in x:
        params = {
            "latitude": i[0],
            "longitude": i[1],
            "start_date": date_string_1,
            "end_date": date_string_2,
            "timezone": "auto",
            "temperature_unit": "fahrenheit",
            "windspeed_unit": "mph",
            "precipitation_unit": "inch",
            "hourly": {"precipitation", "snowfall", "temperature_2m", 
                       "relativehumidity_2m", "surface_pressure", "windspeed_10m",
                       "winddirection_10m", "windgusts_10m", "cloudcover"},
            "daily": {"sunrise", "sunset"},
        }

        response = requests.get("https://archive-api.open-meteo.com/v1/era5", params=params)
        if response.status_code != 200:
            raise Exception(f"API call failed with status code {response.status_code}")
        res = response.json()
        
        # Transformation en DataFrame
        df3 = pd.DataFrame.from_dict(res["daily"], orient="index").T
        df3 = df3.loc[df3.index.repeat(24)].reset_index(drop=True)
        df2 = pd.DataFrame.from_dict(res["hourly"], orient="index").T
        df2 = df2.assign(
            elevation=res["elevation"],
            latitude=res["latitude"],
            longitude=res["longitude"],
            timezone=res["timezone_abbreviation"],
        )
        df2["sunrise"] = df3["sunrise"]
        df2["sunset"] = df3["sunset"]
        li.append(df2)
    return pd.concat(li, axis=0, ignore_index=True)


def api(start_date, end_date, latitude, longitude):
    """api call function.

    Keyword arguments:
    time -- desired date in "%Y-%m-%d format
    latitude -- gps coordinate
    longitude -- gps coordinate
    """

    date_string_1 = start_date + "T" + "00:00"
    date_string_2 = end_date + "T" + "00:00"

    date_object1 = datetime.strptime(date_string_1, "%Y-%m-%dT%H:%M")
    date_object2 = datetime.strptime(date_string_2, "%Y-%m-%dT%H:%M")

    date_only1 = date_object1.date()
    date_only2 = date_object2.date()
    date_string_1 = str(date_only1)
    date_string_2 = str(date_only2)
    # import dataset from API
    x = [(latitude, longitude)]

    li = []
    for i in x:
        params = {
            "latitude": i[0],
            "longitude": i[1],
            "start_date": date_string_1,
            "end_date": date_string_2,
            "timezone": "auto",
            "temperature_unit": "fahrenheit",  #  units
            "windspeed_unit": "mph",
            "precipitation_unit	": "inch",
            "hourly": {
                "precipitation",
                "snowfall",
                "temperature_2m",
                "relativehumidity_2m",
                "surface_pressure",
                "windspeed_10m",
                "winddirection_10m",
                "windgusts_10m",
                "cloudcover",
            },
            "daily": {"sunrise", "sunset"},
        }

        response = requests.get(
            "https://archive-api.open-meteo.com/v1/era5", params=params
        )
        res = response.json()

        df3 = pd.DataFrame.from_dict(res["daily"], orient="index").T
        df3 = df3.loc[df3.index.repeat(24)].reset_index(drop=True)
        df2 = pd.DataFrame.from_dict(res["hourly"], orient="index").T
        df2 = df2.assign(
            elevation=res["elevation"],
            latitude=res["latitude"],
            longitude=res["longitude"],
            timezone=res["timezone_abbreviation"],
        )
        df2["sunrise"] = df3["sunrise"]
        df2["sunset"] = df3["sunset"]
        li.append(df2)
    frame = pd.concat(li, axis=0, ignore_index=True)
    return frame, res



def filtered_df_id(data, id:int):
    """
    Filtrer le DataFrame sur un ID spécifique
    :param data: DataFrame à filtrer
    :param id: ID sur lequel le filtre est appliqué
    """
    try:
        test = data[data['id'] == id]
    except Exception as e:
        print(f"le DataFrame {data} n'a pas pu être filtré avec l'id {id} : {e} ")
    return test


def fetch_weather_data(test_df, api_func):
    """
    Regroupe les données par ID et récupère les données météo pour chaque groupe à l'aide d'une API.

    :param test_df: DataFrame à traiter, contenant les colonnes 'id', 'date', 'latitude' et 'longitude'
    :param api_func: Fonction API à appeler pour obtenir les données météo (doit prendre start_time, end_time, lat, lon comme arguments)
    :return: DataFrame fusionné avec les données météo obtenues
    """
    grouped_conc = test_df.groupby("id")
    frames = []
    
    for id, group in grouped_conc:
        start_time = str(group["date"].iloc[0].date())
        end_time = str(group["date"].iloc[-1].date())
        lat = group["latitude"].iloc[0]
        lon = group["longitude"].iloc[0]
        
        # Appel de l'API pour récupérer les données météo
        frame, res = api_func(start_time, end_time, lat, lon)
        
        frames.append(frame)
    
    # Convertir la colonne 'time' des données météo et 'date' des données initiales au format datetime pour la fusion
    frame = pd.concat(frames, axis=0, ignore_index=True)
    frame['time'] = pd.to_datetime(frame['time'], format='%Y-%m-%dT%H:%M')
    test_df['time'] = pd.to_datetime(test_df['date'], format='%Y-%m-%d %H:%M:%S')
    
    # Filtrer les colonnes utiles dans les données météo pour la fusion
    weather_columns = ['time', 'surface_pressure', 'snowfall', 'temperature_2m',
                       'winddirection_10m', 'relativehumidity_2m', 'windgusts_10m',
                       'windspeed_10m', 'precipitation', 'cloudcover', 'elevation',
                       'timezone', 'sunrise', 'sunset']
    frame = frame[weather_columns]
    
    # Fusionner les données de test avec les données météo sur la colonne 'time'
    df_merged = pd.merge(test_df, frame, on='time', how='left')
    
    return df_merged

     

def predict(data):
    """
    Solar radiation prediction function and weather data extraction.

    Keyword arguments:

    data -- merged hourly data frame from inventory and SBAP using merging fonction
    
    """
    data["snowfall"] = 0
    data["precipitation"] = 0
    data["cloudcover"] = 0

    data["surface_pressure"] = 0
    data["winddirection_10m"] = 0
    data["windgusts_10m"] = 0
    data["windspeed_10m"] = 0
    data["relativehumidity_2m"] = 0
    data["temperature_2m"] = 0
    data["elevation"] = 0
    data["Solar_radiation"] = 0

    devices = []

    grouped_conc = data.groupby("id")

    for id, group in grouped_conc:

        start_time = str(group["date"].iloc[0].date())
        # last time stamp
        end_time = str(group["date"].iloc[len(group) - 1].date())
        lat = group["latitude"].iloc[0]
        lon = group["longitude"].iloc[0]
        frame, res = api(start_time, end_time, lat, lon)
        for i in range(len(group)):

            times = str(group["date"].iloc[i].date())

            hour2 = group["date"].iloc[i].strftime("%H:00")

            snowfall = frame.query("time=='{}'".format(times + "T" + hour2))[
                "snowfall"
            ].values[0]
            precipitation = frame.query("time=='{}'".format(times + "T" + hour2))[
                "precipitation"
            ].values[0]
            cloudcover = frame.query("time=='{}'".format(times + "T" + hour2))[
                "cloudcover"
            ].values[0]

            surface_pressure = frame.query("time=='{}'".format(times + "T" + hour2))[
                "surface_pressure"
            ].values[0]
            winddirection_10m = frame.query("time=='{}'".format(times + "T" + hour2))[
                "winddirection_10m"
            ].values[0]
            windgusts_10m = frame.query("time=='{}'".format(times + "T" + hour2))[
                "windgusts_10m"
            ].values[0]
            windspeed_10m = frame.query("time=='{}'".format(times + "T" + hour2))[
                "windspeed_10m"
            ].values[0]
            relativehumidity_2m = frame.query("time=='{}'".format(times + "T" + hour2))[
                "relativehumidity_2m"
            ].values[0]
            temperature_2m = frame.query("time=='{}'".format(times + "T" + hour2))[
                "temperature_2m"
            ].values[0]
            elevation = frame.query("time=='{}'".format(times + "T" + hour2))[
                "elevation"
            ].values[0]

            group["snowfall"].iloc[i] = snowfall
            group["cloudcover"].iloc[i] = cloudcover

            group["surface_pressure"].iloc[i] = float(surface_pressure)
            group["winddirection_10m"].iloc[i] = winddirection_10m
            group["windspeed_10m"].iloc[i] = float(windspeed_10m)

            group["relativehumidity_2m"].iloc[i] = relativehumidity_2m
            group["temperature_2m"].iloc[i] = float(temperature_2m)
            group["elevation"].iloc[i] = elevation

           

        devices.append(group)

    conc = pd.concat(devices, ignore_index=True)

    return conc




def delete_unnecessary_col(data, columns:list):
    """
    Supprimer les colonnes inutiles
    :param data: le DataFrame concerné
    :param columns: liste des colonnes à supprimer
    
    :return: DataFrame sans les colonnes spécifiées
    """
    data.drop(columns=columns, 
                 axis = 1, 
                 inplace = True )
    
    return data


# FonctioSeparate target variable Y from features X
def separe_target_variables(data):
    """
    Fonction pour séparer les variables de la variable cible
    
    :param data: le DataFrame concerné
    
    return: X: variable DataFrame, Y: target pandas series
    
    """
    target_variable= "GHI"

    X = data.drop(target_variable, axis = 1)
    Y = data.loc[:,target_variable]

    return X, Y



def detect_numeric_and_date_columns(X):
    """
    Fonction pour détecter les colonnes numériques et de date dans un DataFrame.
    
    :param df: DataFrame contenant les données
    
    :return: numeric_features (list): Noms des colonnes numériques
             date_columns (list): Noms des colonnes de dates
    """
    # Initialiser des listes vides pour les colonnes numériques et de dates
    numeric_features = []
    date_columns = []

    # Parcourir les colonnes du DataFrame avec .items()
    for column_name, column_data in X.items():
        if column_name in ['year', 'month', 'day', 'hour', 'weekday']:
            # Ajouter les colonnes de date dans la liste correspondante
            date_columns.append(column_name)
        elif pd.api.types.is_numeric_dtype(column_data):
            # Ajouter les colonnes numériques dans la liste correspondante
            numeric_features.append(column_name)
    
    return numeric_features, date_columns


# Create pipeline for numeric features
def create_pipeline(numeric_features:list, date_columns:list):
    """
    Créer un pipeeline de données
    
    :param date_columns : liste des noms de colonnes des dates
    :param numeric_features : liste des noms de colonne des variables numériques
    
    """
    numeric_transformer = Pipeline(steps=[
        #('imputer', SimpleImputer(strategy='mean')) # missing values will be replaced by columns' mean
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features), # Appliquer le transformer aux colonnes numériques
            ("pass", "passthrough", date_columns),  # Passer les colonnes de dates sans transformation
           ]
    )

    return preprocessor

def split_train_test_temporal(X, Y):
    """
    Séparer les ensembles X et Y en ensemble train & test (80%) tout en conservant la séquence de temporalité
    
    :param X : DataFrame avec les variables explicative
    :param y : Pandas series avec les variables cibles
    
    return : 
    
    """    
    # Define size train % (80 % des données)
    train_size = int(0.8 * len(X))

    # Separate manually train set and test set
    X_train = X[:train_size]  
    X_test = X[train_size:]   
    y_train = Y[:train_size]  
    y_test = Y[train_size:] 
    
    return X_train, X_test, y_train, y_test 



def apply_preprocessor(X_train, X_test, preprocessor):
    
    # Apply preprocessor on X_train
    X_train = preprocessor.fit_transform(X_train)  # fit et transform sur le train set
    print('...Done.')
    print(X_train[0:5])  # Afficher les 5 premières lignes (Numpy array)


    # Preprocessing du test set
    print("Performing preprocessings on test set...")
    print(X_test.head())

    # Apply same preprocessor (sans refit) on X_test
    X_test = preprocessor.transform(X_test)  # On transforme le test set avec les paramètres ajustés sur le train set
    print('...Done.')
    print(X_test[0:5,:])  # Afficher les 5 premières lignes (Numpy array)


    return X_train, X_test


def evaluate_model_combinations(X_train, X_test, y_train, y_test, colonnes_fixes, colonnes_sup, modeles):
    """
    Évalue toutes les combinaisons de colonnes supplémentaires avec les modèles fournis.
    
    Parameters:
        X_train (np.array): Array des features d'entraînement.
        X_test (np.array): Array des features de test.
        y_train (np.array): Array des cibles d'entraînement.
        y_test (np.array): Array des cibles de test.
        colonnes_fixes (list): Liste des indices des colonnes fixes.
        colonnes_sup (list): Liste des indices des colonnes supplémentaires.
        modeles (dict): Dictionnaire des modèles à évaluer avec leur nom.
        
    Returns:
        pd.DataFrame: DataFrame contenant les métriques pour chaque combinaison de colonnes et de modèle.
    """
    
    # Stocker les résultats
    resultats = []
    
    # Générer toutes les combinaisons des colonnes supplémentaires
    for r in range(1, len(colonnes_sup) + 1):
        combinaisons = combinations(colonnes_sup, r)
        
        # Itérer sur chaque combinaison
        for combinaison in combinaisons:
            # Colonnes à utiliser (colonnes fixes + la combinaison actuelle des colonnes supplémentaires)
            colonnes_utilisees = colonnes_fixes + list(combinaison)
            
            # Extraire les colonnes correspondantes du numpy array (utiliser l'indexation numpy [])
            X_train_subset = X_train[:, colonnes_utilisees]
            X_test_subset = X_test[:, colonnes_utilisees]
            
            # Dictionnaire pour stocker les métriques pour cette combinaison
            metriques_combinaison = {
                'colonnes_utilisées': colonnes_utilisees
            }
            
            # Itérer sur les modèles
            for nom_modele, modele in modeles.items():
                # Entraîner le modèle
                modele.fit(X_train_subset, y_train)
                
                # Faire des prédictions sur train et test
                y_train_pred = modele.predict(X_train_subset)
                y_test_pred = modele.predict(X_test_subset)
                
                # Calculer les métriques pour le test set
                mse_test = mean_squared_error(y_test, y_test_pred)
                r2_test = r2_score(y_test, y_test_pred)
                mae_test = mean_absolute_error(y_test, y_test_pred)
                
                # Calculer le R² pour le train set
                r2_train = r2_score(y_train, y_train_pred)
                
                # Stocker les résultats pour ce modèle
                metriques_combinaison[f'mse_test_{nom_modele}'] = mse_test
                metriques_combinaison[f'mae_test_{nom_modele}'] = mae_test
                metriques_combinaison[f'r2_test_{nom_modele}'] = r2_test
                metriques_combinaison[f'r2_train_{nom_modele}'] = r2_train
            
            # Ajouter les résultats pour cette combinaison
            resultats.append(metriques_combinaison)
    
    # Convertir les résultats en DataFrame pour une comparaison facile
    df_resultats = pd.DataFrame(resultats)
    return df_resultats

def find_best_combination_and_model(df_resultats):
    """
    Trouve la meilleure combinaison de colonnes et le modèle ayant le score r2_test le plus élevé.

    :param df_resultats: DataFrame contenant les résultats des différents modèles et combinaisons de colonnes.
    :return: Un dictionnaire avec la meilleure combinaison de colonnes, le nom du modèle et le score r2_test maximum.
    """
    # Filtrer les colonnes contenant 'r2_test_' pour chaque modèle
    r2_test_columns = df_resultats.filter(like='r2_test_')

    # Identifier le modèle et la combinaison ayant le r2_test le plus élevé
    max_r2_column = r2_test_columns.max().idxmax()  # La colonne du modèle avec le r2_test maximum
    max_r2_value = r2_test_columns[max_r2_column].max()  # La valeur maximum de r2_test
    best_result = df_resultats[r2_test_columns[max_r2_column] == max_r2_value]

    # Extraire la meilleure combinaison de colonnes et le modèle associé
    best_combination = best_result['colonnes_utilisées'].values[0]
    best_model_name = max_r2_column.replace('r2_test_', '')  # Nom du modèle avec le meilleur r2_test

    # Retourner les trois valeurs
    return best_combination, best_model_name, max_r2_value

def select_optimize_col_pred(X_train, X_test, columns_to_keep):
    """
    filtrer le Dataframe avec les colonnes qui optimisent les résultats de prédictions
    
    :param X_train : échantillon d'entraînement (numpy array, 80% des données)
    :param X_test : ééchantillon de test (numpy array, 20% des données)

    return: 
    """
    # Les indices des colonnes gardées
    columns_to_keep = columns_to_keep

    # Sélectionner ces colonnes dans X_train (qui est un numpy array)
    X_train_opti = X_train[:, columns_to_keep]
    X_test_opti = X_test[:, columns_to_keep]
    
    return X_train_opti, X_test_opti


def save_model_locally(model, model_name="best_model.joblib"):
    """
    Enregistre le modèle localement dans le volume persistant /usr/local/airflow/models.
    
    :param model: Le modèle entraîné à sauvegarder.
    :param model_name: Nom du fichier pour le modèle. Par défaut "best_model.joblib".
    :return: Chemin où le modèle a été sauvegardé.
    """
    model_dir = "/usr/local/airflow/models"
    os.makedirs(model_dir, exist_ok=True)  # Crée le répertoire s'il n'existe pas
    model_path = os.path.join(model_dir, model_name)
    joblib.dump(model, model_path)
    logging.info(f"Modèle enregistré localement dans {model_path}")
    return model_path

def save_predictions_locally(predictions, filename="predictions.csv"):
    """
    Enregistre les prédictions localement dans le volume persistant /usr/local/airflow/predictions.
    
    :param predictions: Les prédictions à sauvegarder (numpy array ou DataFrame).
    :param filename: Nom du fichier pour les prédictions. Par défaut "predictions.csv".
    :return: Chemin où les prédictions ont été sauvegardées.
    """
    predictions_dir = "/usr/local/airflow/predictions"
    os.makedirs(predictions_dir, exist_ok=True)  # Crée le répertoire s'il n'existe pas
    predictions_path = os.path.join(predictions_dir, filename)
    pd.DataFrame(predictions).to_csv(predictions_path, index=False)
    logging.info(f"Prédictions enregistrées localement dans {predictions_path}")
    return predictions_path

def upload_file_to_s3(local_path: str, bucket_name: str, s3_key: str, aws_conn_id: str = 'aws_s3_conn'):
    """
    Téléverse un fichier spécifique depuis un emplacement local vers un bucket S3.

    Parameters:
        local_path (str): Le chemin local du fichier à téléverser.
        bucket_name (str): Le nom du bucket S3.
        s3_key (str): Le chemin de destination dans S3 (clé S3).
        aws_conn_id (str): L'identifiant de la connexion AWS dans Airflow.
    """
    hook = S3Hook(aws_conn_id=aws_conn_id)
    hook.load_file(filename=local_path, key=s3_key, bucket_name=bucket_name, replace=True)
    print(f"Fichier {local_path} téléversé vers le bucket {bucket_name} sous la clé {s3_key}")


    
