from airflow import DAG
from airflow.providers.http.hooks.http import HttpHook
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.decorators import task
from airflow.utils.dates import days_ago
import pandas as pd
from datetime import timedelta
from sqlalchemy import create_engine
import sys
import os

# Add the project root directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.inference import WeatherPredictor


# Latitude and longitude for the desired location (London in this case)
LATITUDE = '51.5074'
LONGITUDE = '-0.1278'
POSTGRES_CONN_ID='postgres_default'
API_CONN_ID='open_meteo_api'

default_args={
    'owner':'airflow',
    'start_date':days_ago(1)
}

## DAG
with DAG(dag_id='weather_etl_pipeline_lstm',
         default_args=default_args,
         schedule_interval=timedelta(seconds=15),
         catchup=False) as dags:
    
    @task()
    def extract_weather_data():
        """Extract weather data from Open-Meteo API using Airflow Connection."""

        # Use HTTP Hook to get connection details from Airflow connection

        http_hook=HttpHook(http_conn_id=API_CONN_ID,method='GET')

        ## Build the API endpoint
        ## https://api.open-meteo.com/v1/forecast?latitude=51.5074&longitude=-0.1278&current_weather=true
        endpoint=f'v1/forecast?latitude={LATITUDE}&longitude={LONGITUDE}&current=temperature_2m,relative_humidity_2m,precipitation,snowfall,cloud_cover,surface_pressure,wind_speed_10m,wind_direction_10m,wind_gusts_10m&hourly=temperature_2m,snowfall,surface_pressure,cloud_cover&daily=sunrise,sunset'

        ## Make the request via the HTTP Hook
        response=http_hook.run(endpoint)

        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to fetch weather data: {response.status_code}")
        
    @task()
    def transform_weather_data(weather_data):
        """Transform the extracted weather data."""
        current_weather = weather_data['current']
        sun_data = weather_data['daily']
        print(current_weather)
        transformed_data = {
            'latitude': LATITUDE,
            'longitude': LONGITUDE,
            'date': current_weather['time'],
            'surface_pressure': current_weather.get('surface_pressure', 0),
            'snowfall': current_weather.get('snowfall', 0),
            'temperature_2m': current_weather['temperature_2m'],
            'winddirection_10m': current_weather['wind_direction_10m'],
            'relativehumidity_2m': current_weather['relative_humidity_2m'],
            'windgusts_10m': current_weather['wind_gusts_10m'],
            'windspeed_10m': current_weather['wind_speed_10m'],
            'precipitation': current_weather.get('precipitation', 0),
            'cloudcover': current_weather.get('cloud_cover', 0),
            'sunrise': sun_data['sunrise'][0],
            'sunset': sun_data['sunset'][0],
            
        }
        return transformed_data
    
    @task()
    def load_weather_data(transformed_data):
        """Load transformed data into PostgreSQL."""
        pg_hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)
        conn = pg_hook.get_conn()
        cursor = conn.cursor()
        query = "SELECT ghi, date FROM weather_dataset ORDER BY date DESC LIMIT 1;"
        engine = create_engine('postgresql://postgres:postgres@localhost:5555/postgres')

        with engine.connect() as connection:
            new_data = pd.read_sql_query(query, connection)
            new_data = pd.DataFrame(new_data, index=[0])
            new_data = new_data.drop(columns=['ghi'])


        model_path = 'saved_models/lstm_model.pickle'

        #db_url = 'postgresql://postgres:postgres@localhost:5432/postgres'
        weather_predictor = WeatherPredictor(model_path)
        #create a dataframe from the transformed data
        df = pd.DataFrame(transformed_data, index=[0])
        combined_data = pd.concat([new_data, df], ignore_index=True)
        # Optionally, drop the original `date` column if no longer needed
        combined_data = combined_data.drop(columns=['longitude', 'latitude'])
        predictions = weather_predictor.predixt_lstm(combined_data)

        # Create database if it doesn't exist
        #cursor.execute("CREATE DATABASE IF NOT EXISTS postgres;")
        # Create table if it doesn't exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS weather_dataset_lstm (
                latitude FLOAT,
                longitude FLOAT,
                date TIMESTAMP,
                surface_pressure FLOAT DEFAULT 0,
                snowfall FLOAT DEFAULT 0,
                temperature_2m FLOAT,
                winddirection_10m FLOAT,
                relativehumidity_2m FLOAT,
                windgusts_10m FLOAT,
                windspeed_10m FLOAT,
                precipitation FLOAT DEFAULT 0,
                cloudcover FLOAT DEFAULT 0,
                sunrise TIMESTAMP,
                sunset TIMESTAMP,
                ghi FLOAT
            );

        """)

        # Insert transformed data into the table
        cursor.execute("""
            INSERT INTO weather_dataset (
                latitude, longitude, date, surface_pressure, snowfall, temperature_2m, 
                winddirection_10m, relativehumidity_2m, windgusts_10m, windspeed_10m, 
                precipitation, cloudcover, sunrise, sunset, ghi
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                transformed_data['latitude'],
                transformed_data['longitude'],
                transformed_data['date'],
                transformed_data.get('surface_pressure', 0),
                transformed_data.get('snowfall', 0),
                transformed_data['temperature_2m'],
                transformed_data['winddirection_10m'],
                transformed_data['relativehumidity_2m'],
                transformed_data['windgusts_10m'],
                transformed_data['windspeed_10m'],
                transformed_data.get('precipitation', 0),
                transformed_data.get('cloudcover', 0),
                transformed_data['sunrise'],
                transformed_data['sunset'],
                predictions[0]
            ))


        conn.commit()
        cursor.close()

    ## DAG Worflow- ETL Pipeline
    weather_data= extract_weather_data()
    transformed_data=transform_weather_data(weather_data)
    load_weather_data(transformed_data)

        
    




    

