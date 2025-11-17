import streamlit as st
import pandas as pd
import numpy as np
import time
from sqlalchemy import create_engine

# Set up your database connection
engine = create_engine('postgresql://postgres:postgres@localhost:5555/postgres')


# Test the connection
try:
    with engine.connect() as connection:
        result = connection.execute("SELECT 1")
        print("Connection successful!")
except Exception as e:
    print(f"Error connecting to the database: {e}")

# Fetch data function
def fetch_data():
    try:
        query = "SELECT ghi, date FROM weather_dataset ORDER BY date DESC LIMIT 500;"
        with engine.connect() as connection:
            new_data = pd.read_sql_query(query, connection)
        
        # Debug: Check if new data is fetched
        if new_data.empty:
            print("No data returned from the database.")
        else:
            print(f"Fetched {len(new_data)} rows from the database.")
        
        return new_data
    except Exception as e:
        print(f"Error fetching data from the database: {e}")
        return pd.DataFrame()  # Return empty DataFrame in case of error

# Streamlit page setup
st.title("Real-time GHI Monitoring Simulation")

# Initialize session state if not present
if 'last_fetch_time' not in st.session_state:
    st.session_state.last_fetch_time = time.time()  # Initial fetch time
    st.session_state.data = pd.DataFrame(columns=['date', 'GHI'])  # Empty DataFrame to start with

# Create an empty container for real-time data display
chart_container = st.empty()

while True:
    # Fetch new data every 15 seconds
    new_data = fetch_data()
    st.session_state.data = new_data  # Update session state with new data

    # Format the timestamp as DD/MM/YYYYTHH:MM
    st.session_state.data['date'] = pd.to_datetime(st.session_state.data['date'])
    st.session_state.data['date'] = st.session_state.data['date'].dt.strftime('%d/%m/%YT%H:%M')

    # Display the new data in a chart
    if not st.session_state.data.empty:
        chart_container.line_chart(st.session_state.data.set_index('date')['ghi'])
    else:
        chart_container.write("No data available for plotting!")

    # Wait for 30 seconds before rerunning the app
    time.sleep(360)
    st.experimental_rerun()  # Forces Streamlit to rerun the script
