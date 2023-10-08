import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

import numpy as np

# reading dataset
df = pd.read_csv("C:\\Users\\Dell\\Downloads\\sihdataset.csv")
# df.head()
# viewing all variables to decide which are not useful
# df.columns
# removing variables not needed for the model
df_clean = df.drop(columns=["massif_num","aval_type", "acccidental_risk_index", 
                            'snow_thickness_1D', 'snow_thickness_3D', 'snow_thickness_5D',
                           'snow_water_1D', 'snow_water_3D', 'snow_water_5D', 'risk_index',
                           'thickness_of_wet_snow_top_of_snowpack','thickness_of_frozen_snow_top_of_snowpack',
                            'rainfall_rate', 'drainage', 'runoff'
                           , 'frozen_water_in_soil','snow_melting_rate'])

df_clean.info()

# Define the features (X) and the target variable (y)
X = df_clean[['elevation',"lon","lat",'temp_soil_0.005_m', 'temp_soil_0.08_m','liquid_water_in_soil', 'whiteness_albedo', 'net_radiation',
        'surface_temperature' ,'surface_air_pressure_mean', 'near_surface_humidity_mean', 'relative_humidity_mean', 
        'freezing_level_altitude_mean', 'rain_snow_transition_altitude_mean', 
        'air_temp_max', 'wind_speed_max', 'snowfall_rate_max', 'nebulosity_max', 
        'air_temp_min', 'aval_accident' ]]

y = df_clean['aval_event']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a Random Forest Classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = clf.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))





feature_labels = [
    "Elevation", "Longitude", "Latitude", "Soil Temp 0.005m", "Soil Temp 0.08m",
    "Liquid Water in Soil", "Whiteness Albedo", "Net Radiation", "Surface Temperature",
    "Near Surface Humidity", "Relative Humidity", "Freezing Altitude", "Rain/Snow Transition Altitude",
    "Max Air Temp", "Max Wind Speed", "Max Snowfall Rate", "Max Nebulosity", "Min Air Temp", "Avalanche Accident"
]

# Define an input array with the same features as your training data


# Print the message





st.title('Avalanche Prediction Model')

# Create a number input box for an integer
Elevation_input =  st.number_input("Enter Elevation:", min_value=None, max_value=None, value=0, step=1, format="%d")
# Longitude
Longitude_input = st.number_input("Enter a Longitude:", min_value=None, max_value=None, value=0.0, step=0.000001, format="%.6f")
# Latitude
Latitude_input = st.number_input("Enter a Latidude:", min_value=None, max_value=None, value=0.0, step=0.000001, format="%.6f")
# soil temp at 0.008m
Soil_Temp_8m = st.number_input("Enter Soil temp at 0.008m", min_value=None, max_value=None, value=273.0, step=0.000001, format="%.6f")
# soil temp at 0.05m
Soil_Temp_5m = st.number_input("Enter Soil temp at 0.05m", min_value=None, max_value=None, value=273.0, step=0.000001, format="%.6f")
# Liquid Water in Soil
Liquid_Water_in_Soil = st.number_input("Enter Liquid water in soil:", min_value=None, max_value=None, value=0.2, step=0.000001, format="%.6f")
# Whiteness Albedo
Whiteness_Albedo = st.number_input("Enter Whiteness Albedo:", min_value=None, max_value=None, value=0.2, step=0.000001, format="%.6f")
# Surface Temperature
Surface_Temperature = st.number_input("Enter Surface Temperature:", min_value=None, max_value=None, value=273.0, step=0.000001, format="%.6f")
# Surface Air Temperature
Surface_Air_Temp = st.number_input("Enter Surface Air Temperature:", min_value=None, max_value=None, value=273.0, step=0.000001, format="%.6f")
# Near Surface Humidity
Near_Surface_Humidity = st.number_input("Enter Surface Humidity:", min_value=None, max_value=None, value=0.006, step=0.001, format="%.6f")
# Relative Humidity
Relative_Humidity = st.number_input("Enter Relative Humidity:", min_value=None, max_value=None, value=60.00, step=0.000001, format="%.6f")
# Freezing Altitude
Freezing_Altitude = st.number_input("Enter Freezing Altitude", min_value=None, max_value=None, value=1000.00, step=0.000001, format="%.6f")
# Rain/Snow Transition Altitude
Rain_or_Snow_Transition_Altitude = st.number_input("Enter Rain or Snow Transition Altitude", min_value=None, max_value=None, value=500.00, step=0.000001, format="%.6f")
# Max Air Temp
Max_Air_Temp = st.number_input("Enter Max Air Temp", min_value=None, max_value=None, value=0.0, step=0.000001, format="%.6f") 
# Max Wind Speed
Max_Wind_Speed = st.number_input("Enter Max Wind Speed", min_value=None, max_value=None, value=4.000, step=0.000001, format="%.6f") 
# Max Snowfall Rate
Max_Snowfall_Rate = st.number_input("Enter Max Snow Fall Rate", min_value=None, max_value=None, value=0.0, step=0.0, format="%.6f")
# Max Nebulosity
Max_Nebulosity = st.number_input("Enter Max Nebulosity", min_value=None, max_value=None, value=0.9, step=0.000001, format="%.6f")
# Min Air Temp
Min_Air_Temp =  st.number_input("Enter Max Air Temp", min_value=None, max_value=None, value=280.000, step=0.000001, format="%.6f")
# Avalanche Accident (0 or 1)
Avalanche_Accident = st.number_input("Enter Avalanche Accident", min_value=None, max_value=None, value=0.0, step=0.000001, format="%.6f")
# Net Radiation
Net_Radiation = st.slider("Set Net Radiation", min_value=0.00, max_value=100.00, value=50.00, step=0.01, format="%.4f")


def predictFunc(Elevation_input, Longitude_input, Latitude_input, Soil_Temp_8m, Soil_Temp_5m, Liquid_Water_in_Soil, Whiteness_Albedo, Surface_Temperature,
                Surface_Air_Temp, Near_Surface_Humidity, Relative_Humidity, Freezing_Altitude, Rain_or_Snow_Transition_Altitude, Max_Air_Temp, Max_Wind_Speed,
                Max_Snowfall_Rate, Max_Nebulosity, Min_Air_Temp, Avalanche_Accident, Net_Radiation):

    input_data = np.array([
        [Elevation_input, Longitude_input, Latitude_input, Soil_Temp_8m, Soil_Temp_5m, Liquid_Water_in_Soil, Whiteness_Albedo, Surface_Temperature,
                Surface_Air_Temp, Near_Surface_Humidity, Relative_Humidity, Freezing_Altitude, Rain_or_Snow_Transition_Altitude, Max_Air_Temp, Max_Wind_Speed,
                Max_Snowfall_Rate, Max_Nebulosity, Min_Air_Temp, Avalanche_Accident, Net_Radiation]
    ])

    # Make predictions using the trained model
    predictions = clf.predict(input_data)

    # Define a message based on the prediction
    if predictions[0] == 0:
        message = "There is a lower chance of an avalanche event."
    else:
        message = "There is a higher chance of an avalanche event."

    st.write(message)

if st.button("Predict Avalanche"):
    predictFunc(Elevation_input, Longitude_input, Latitude_input, Soil_Temp_8m, Soil_Temp_5m, Liquid_Water_in_Soil, Whiteness_Albedo, Surface_Temperature,
                Surface_Air_Temp, Near_Surface_Humidity, Relative_Humidity, Freezing_Altitude, Rain_or_Snow_Transition_Altitude, Max_Air_Temp, Max_Wind_Speed,
                Max_Snowfall_Rate, Max_Nebulosity, Min_Air_Temp, Avalanche_Accident, Net_Radiation)

