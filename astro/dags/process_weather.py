import pandas as pd
from datetime import datetime, timedelta
from airflow.decorators import task_group, task, dag
from src.functions import *  # Import des fonctions
import logging

# Paramètres par défaut pour le DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=200),
}

@dag(
    dag_id='data_pipeline_solar_pred',
    default_args=default_args,
    description='A data pipeline for solar prediction with parallel file loading',
    schedule_interval='0 8 * * 1',  # Exécution chaque lundi à 8h
    start_date=datetime(2024, 10, 31),
    catchup=False,
    dagrun_timeout=timedelta(hours=2)
)
def solar_prediction_dag():
    
    # Importer les datasets
    @task_group(group_id='import_data')
    def import_data():
        
        @task(task_id='download_energy')
        def download_energy():
            energy_path = '/usr/local/airflow/data/energy_data.csv'  # Volume partagé
            download_file_from_s3(
                bucket_name='datalake-s3-data',
                s3_key='Solar_Energy_Production.csv',
                local_path=energy_path,
                aws_conn_id = 'aws_s3_conn'
            )
            return {'energy_path': energy_path}

        @task(task_id='download_photovoltaic')
        def download_photovoltaic():
            photovoltaic_path = '/usr/local/airflow/data/photov_data.csv'  # Volume partagé
            download_file_from_s3(
                bucket_name='datalake-s3-data',
                s3_key='Solar_Photovoltaic_Sites.csv',
                local_path=photovoltaic_path,
                aws_conn_id = 'aws_s3_conn'
            )
            return {'photov_path': photovoltaic_path}
        
        energy_data = download_energy()
        photovoltaic_data = download_photovoltaic()
        
        return energy_data, photovoltaic_data

    @task_group(group_id='filter_columns')
    def filter_columns(energy_data, photovoltaic_data):
        
        @task(task_id='filter_energy')
        def filter_energy(data):
            df = pd.read_csv(data['energy_path'])
            cols_to_keep = ['name', 'id', 'address', 'date', 'kWh']
            
            filtered_df = keep_useful_columns(df, cols_to_keep)
            filtered_energy_path = '/usr/local/airflow/data/filtered_energy.csv'  # Volume partagé
            filtered_df.to_csv(filtered_energy_path, index=False)
            return {'filtered_energy_path': filtered_energy_path}
        
        @task(task_id='filter_photovoltaic')
        def filter_photovoltaic(data):
            df = pd.read_csv(data['photov_path'])
            cols_to_keep = ['id', 'latitude', 'longitude']
            
            filtered_df = keep_useful_columns(df, cols_to_keep)
            filtered_photovoltaic_path = '/usr/local/airflow/data/filtered_photov.csv'  # Volume partagé
            filtered_df.to_csv(filtered_photovoltaic_path, index=False)
            return {'filtered_photov_path': filtered_photovoltaic_path}
        
        filtered_energy = filter_energy(energy_data)
        filtered_photovoltaic = filter_photovoltaic(photovoltaic_data)
        
        return filtered_energy, filtered_photovoltaic
    
    @task(task_id='merge_data')
    def merge_data(filtered_energy, filtered_photovoltaic):
        # Charger les fichiers CSV sélectionnés
        df_energy = pd.read_csv(filtered_energy['filtered_energy_path'])
        df_photovoltaic = pd.read_csv(filtered_photovoltaic['filtered_photov_path'])
        
        # Fusionner les deux DataFrames en fonction de la clé commune 'id'
        df_merged = merge_dataframes(df_energy, df_photovoltaic, key='id', how='left')
        
        # Enregistrer le DataFrame fusionné
        merged_data_path = '/usr/local/airflow/data/merged_data.csv'  # Volume partagé
        df_merged.to_csv(merged_data_path, index=False)
        return merged_data_path
    
    @task(task_id='process_merge_data')
    def process(merged_data_path):
        df_merged = pd.read_csv(merged_data_path)
        # Traiter le DataFrame fusionné
        df_merged = process_merged_dataframe(df_merged)
        # Convertir la colonne 'date' au format datetime
        df_merged = convert_datetime(df_merged, 'date')
        # Renommer la colonne 'kWh' en 'GHI' et trier les colonnes
        df_merged = rename_and_sort(df_merged, 'kWh', 'GHI', ['name', 'date'])
        # Extraire les caractéristiques temporelles depuis la colonne 'date'
        df_merged = extract_temporality(df_merged, 'date')
        # Enregistrer le DataFrame fusionné
        merged_data_process_path = '/usr/local/airflow/data/merged_data.csv'  # Volume partagé
        df_merged.to_csv(merged_data_process_path, index=False)
        return merged_data_process_path
    
    @task(task_id='api_call', execution_timeout = timedelta(hours=1))
    def api_call_predict(merged_data_process_path):
        # Charger le fichier de données pré-traité
        df_merged_process = pd.read_csv(merged_data_process_path)
        
        start_time = str('2023-12-04')
        end_time = str('2024-01-04')
        lat, lon = 46.999873, 6.498147
        res = api(start_time, end_time, lat, lon)
        # convertir la colonne date au format datetime
        df_merged_process['date'] = pd.to_datetime(df_merged_process['date'])
        # Enregistrer le type de la colonne 'date' dans les logs
        logging.info(f"Type de la colonne 'date' après conversion : {df_merged_process['date'].dtype}")
        # Définir un ID spécifique pour limiter la durée de l'appel API
        specific_id = 577650  # Remplacez par l'ID souhaité
        # Filtrer le DataFrame pour cet ID spécifique
        df_id = filtered_df_id(df_merged_process, specific_id)
        # Récupérer et fusionner les données météo en utilisant fetch_weather_data
        # En passant api_new comme argument pour les appels API spécifiques
        df_id_merged = fetch_weather_data(df_id, api)
        # Obtenir les prédictions de rayonnement solaire
        df_resp_api = predict(df_id_merged)
        
        # Enregistrer le résultat final dans un fichier CSV
        resp_api_path = '/usr/local/airflow/data/final_resp_data.csv'
        df_resp_api.to_csv(resp_api_path, index=False)
        return resp_api_path
    
    @task(task_id = 'preprocessing_before_model')
    def preprocessing_to_apply_model(resp_api_path):
        df_resp_api = pd.read_csv(resp_api_path)
        # Supprimer les colonnes inutiles

        unnecessary_col = ['Solar_radiation', 'latitude', 'longitude', 'id', 'name', 'address', 'elevation', "precipitation", 
             "date", 'sunrise', 'sunset', 'timezone', 'time', 'windgusts_10m']
        df_resp_api = delete_unnecessary_col(df_resp_api, unnecessary_col)

        # Séparer les variables explicatives (X) de la variable cible (Y)
        X, Y = separe_target_variables(df_resp_api)
        # Identifier les colonnes numériques et de dates dans X
        numeric_features, date_columns = detect_numeric_and_date_columns(X)
        # Créer le pipeline de prétraitement
        preprocessor = create_pipeline(numeric_features, date_columns)
        # Séparer les ensembles d'entraînement et de test en respectant la séquence temporelle
        X_train, X_test, y_train, y_test = split_train_test_temporal(X, Y)
        # Instancier la fonction afin d'appliquer le preprocessor
        X_train, X_test = apply_preprocessor(X_train, X_test, preprocessor)
        
        # Sauvegarder les numpy arrays en CSV
        X_train_path = '/usr/local/airflow/data/X_train.csv'
        X_test_path = '/usr/local/airflow/data/X_test.csv'
        y_train_path = '/usr/local/airflow/data/y_train.csv'
        y_test_path = '/usr/local/airflow/data/y_test.csv'
        
        pd.DataFrame(X_train).to_csv(X_train_path, index=False)
        pd.DataFrame(X_test).to_csv(X_test_path, index=False)
        pd.DataFrame(y_train).to_csv(y_train_path, index=False)
        pd.DataFrame(y_test).to_csv(y_test_path, index=False)

        # Retourner les chemins vers les fichiers
        return {
            'X_train_path': X_train_path,
            'X_test_path': X_test_path,
            'y_train_path': y_train_path,
            'y_test_path': y_test_path
        }

    @task(task_id='choose_model_train_and_predict', execution_timeout = timedelta(hours=1))
    def train_model_predict(file_paths):
        
        
        # Charger les fichiers CSV en numpy arrays ou DataFrames
        X_train = pd.read_csv(file_paths['X_train_path']).values
        X_test = pd.read_csv(file_paths['X_test_path']).values
        y_train = pd.read_csv(file_paths['y_train_path']).values
        y_test = pd.read_csv(file_paths['y_test_path']).values
        
         # Définir les colonnes fixes : les 5 premières colonnes sont temporelles
        colonnes_fixes = [0, 1, 2, 3, 4]  # year, month, day, hour, weekday
        # Colonnes supplémentaires (les colonnes 5, 6, 7, 8, 9, 10 dans cet exemple)
        colonnes_sup = [5, 6, 7, 8, 9, 10, 11]  # temperature, humidity, wind_speed, etc.
        # Initialiser les modèles dans un dictionnaire
        modeles = {
            "XGBoost": XGBRegressor(random_state = 42),
            "Gradient_boosting-regressor": GradientBoostingRegressor(random_state = 42),
            "Linear_regression": LinearRegression(),
            "RandomForest": RandomForestRegressor(random_state = 42)
        }
        
        
        df_resultats = evaluate_model_combinations(X_train, X_test, y_train, y_test, colonnes_fixes, colonnes_sup, modeles)  
         
        # Filtrer les colonnes contenant 'r2_test_' pour chaque modèle
        best_combination, best_model_name, max_r2_value = find_best_combination_and_model(df_resultats)

        # Afficher les résultats
        logging.info(f"Meilleure combinaison de colonnes : {best_combination}")
        logging.info(f"Modèle avec le meilleur r2_test : {best_model_name}")
        logging.info(f"Score r2_test maximum : {max_r2_value}")

        # Créer X_train et X_test avec la best_combinaison de colonne
        X_train_opti, X_test_opti = select_optimize_col_pred(X_train, X_test, best_combination)
        # Configuration des paramètres pour GridSearchCV
    
        param_grid = {'n_estimators': [220], 
                      'learning_rate': [0.01],  
                      'max_depth': [7, 10], 
                      'min_samples_split': [6, 7, 8], 
                      'min_samples_leaf': [6, 7],
                      'subsample': [0.6]}

        # Utiliser TimeSeriesSplit pour respecter l'ordre des séries temporelles
        tscv = TimeSeriesSplit(n_splits=5)
        # Initialiser le modèle
        gbr = GradientBoostingRegressor()
        # Configurer la recherche par grille
        grid_search = GridSearchCV(estimator=gbr, param_grid=param_grid, cv=tscv)
        # Effectuer la recherche par grille sur les données d'entraînement
        grid_search.fit(X_train_opti, y_train)
        # Récupérer le meilleur modèle
        best_model = grid_search.best_estimator_
        y_train_pred = best_model.predict(X_train_opti)
        y_test_pred = best_model.predict(X_test_opti)
        r2_train = r2_score(y_train, y_train_pred)
        r2_test = r2_score(y_test, y_test_pred)
        logging.info(f"r² Score train : {r2_train}")
        logging.info(f"r² score test : {r2_test}")
        
        
        # Sauvegarder le modèle localement
        model_path = save_model_locally(best_model, "best_model.joblib")
        # Sauvegarder les prédictions localement
        predictions_path = save_predictions_locally(y_test_pred, "predictions.csv")

        return {
            'model_path': model_path,
            'predictions_path': predictions_path
        }
    
    @task(task_id='upload_to_s3')   
    def upload_model_and_predictions_to_s3(paths):
        # Définir le bucket et les clés S3 pour le modèle et les prédictions
        bucket_name = "datalake-s3-data"
        model_s3_key = "best_model.joblib"
        predictions_s3_key = "predictions.csv"
        aws_conn_id = 'aws_s3_conn'
        
        # upload le modèle et les prédictions sur S3
        upload_file_to_s3(paths['model_path'], bucket_name=bucket_name, s3_key=model_s3_key, aws_conn_id=aws_conn_id)
        upload_file_to_s3(paths['predictions_path'], bucket_name=bucket_name, s3_key=predictions_s3_key, aws_conn_id=aws_conn_id)
    
    logging.info("Modèle et prédictions téléversés sur S3 avec succès.")
            
    # Appel des groupes de tâches et enchaînement
    imported_data = import_data()
    filtered_data = filter_columns(imported_data[0], imported_data[1])
    merged_data_path = merge_data(filtered_data[0], filtered_data[1])
    process_data_path = process(merged_data_path)
    resp_api_path = api_call_predict(process_data_path)
    file_paths = preprocessing_to_apply_model(resp_api_path)
    paths = train_model_predict(file_paths)
    upload_model_and_predictions_to_s3(paths)
    
        
    
    
# Créer une instance de DAG
dag = solar_prediction_dag()
