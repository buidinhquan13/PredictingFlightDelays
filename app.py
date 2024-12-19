import streamlit as st
import pandas as pd
import joblib
import folium
import requests

from datetime import datetime, timedelta, date

#st.title("🚗 FLIGHT DELAY PREDICTION 🚗")


airport_df = pd.read_csv('airport_coordinates.csv')
origin_airport_options = airport_df['origin_airport'].dropna().unique().tolist()
origin_airport_options.sort()

destination_airport_options = airport_df['destination_airport'].dropna().unique().tolist()
destination_airport_options.sort()

# Load the pre-trained model
model = joblib.load('CatBoost_model.pkl')
    
    
## Dictionary for carrier code
carrier_names = ["American Airlines (AA)", "Alaska Airlines (AS)", "JetBlue Airways (B6)", "Delta Air Lines (DL)", "Frontier Airlines (F9)", 
                 "Allegiant Air (G4)", "Hawaiian Airlines (HA)", "Spirit Airlines (NK)", "United Airlines (UA)", "Southwest Airlines (WN)"]
carrier_code_dict = {
    "American Airlines (AA)": 0,
    "Alaska Airlines (AS)": 1,
    "JetBlue Airways (B6)": 2,
    "Delta Air Lines (DL)": 3,
    "Frontier Airlines (F9)": 4,
    "Allegiant Air (G4)": 5,
    "Hawaiian Airlines (HA)": 6,
    "Spirit Airlines (NK)": 7,
    "United Airlines (UA)": 8,
    "Southwest Airlines (WN)": 9
}

airport_dict = {
    'ABE': 0, 'ABI': 1, 'ABQ': 2, 'ABR': 3, 'ABY': 4, 'ACK': 5, 'ACT': 6, 'ACV': 7, 
    'ACY': 8, 'ADK': 9, 'ADQ': 10, 'AEX': 11, 'AGS': 12, 'AKN': 13, 'ALB': 14, 'ALO': 15, 
    'ALW': 16, 'AMA': 17, 'ANC': 18, 'APN': 19, 'ART': 20, 'ASE': 21, 'ATL': 22, 'ATW': 23, 
    'ATY': 24, 'AUS': 25, 'AVL': 26, 'AVP': 27, 'AZA': 28, 'AZO': 29, 'BDL': 30, 'BET': 31, 
    'BFF': 32, 'BFL': 33, 'BFM': 34, 'BGM': 35, 'BGR': 36, 'BHM': 37, 'BIL': 38, 'BIS': 39, 
    'BJI': 40, 'BKG': 41, 'BLI': 42, 'BLV': 43, 'BMI': 44, 'BNA': 45, 'BOI': 46, 'BOS': 47, 
    'BPT': 48, 'BQK': 49, 'BRD': 50, 'BRO': 51, 'BRW': 52, 'BTM': 53, 'BTR': 54, 'BTV': 55, 
    'BUF': 56, 'BUR': 57, 'BWI': 58, 'BZN': 59, 'CAE': 60, 'CAK': 61, 'CDC': 62, 'CDV': 63, 
    'CGI': 64, 'CHA': 65, 'CHO': 66, 'CHS': 67, 'CID': 68, 'CIU': 69, 'CKB': 70, 'CLE': 71, 
    'CLL': 72, 'CLT': 73, 'CMH': 74, 'CMI': 75, 'CMX': 76, 'CNY': 77, 'COD': 78, 'COS': 79, 
    'COU': 80, 'CPR': 81, 'CRP': 82, 'CRW': 83, 'CSG': 84, 'CVG': 85, 'CWA': 86, 'CYS': 87, 
    'DAB': 88, 'DAL': 89, 'DAY': 90, 'DBQ': 91, 'DCA': 92, 'DEN': 93, 'DFW': 94, 'DHN': 95, 
    'DIK': 96, 'DLG': 97, 'DLH': 98, 'DRO': 99, 'DRT': 100, 'DSM': 101, 'DTW': 102, 'DUT': 103, 
    'DVL': 104, 'EAR': 105, 'EAT': 106, 'EAU': 107, 'ECP': 108, 'EGE': 109, 'EKO': 110, 'ELM': 111, 
    'ELP': 112, 'ERI': 113, 'ESC': 114, 'EUG': 115, 'EVV': 116, 'EWN': 117, 'EWR': 118, 'EYW': 119, 
    'FAI': 120, 'FAR': 121, 'FAT': 122, 'FAY': 123, 'FCA': 124, 'FLG': 125, 'FLL': 126, 'FLO': 127, 
    'FNT': 128, 'FSD': 129, 'FSM': 130, 'FWA': 131, 'GCC': 132, 'GCK': 133, 'GEG': 134, 'GFK': 135, 
    'GGG': 136, 'GJT': 137, 'GNV': 138, 'GPT': 139, 'GRB': 140, 'GRI': 141, 'GRK': 142, 'GRR': 143, 
    'GSO': 144, 'GSP': 145, 'GTF': 146, 'GTR': 147, 'GUC': 148, 'GUM': 149, 'HDN': 150, 'HGR': 151, 
    'HIB': 152, 'HLN': 153, 'HNL': 154, 'HOB': 155, 'HOU': 156, 'HPN': 157, 'HRL': 158, 'HSV': 159, 
    'HTS': 160, 'HVN': 161, 'HYA': 162, 'HYS': 163, 'IAD': 164, 'IAG': 165, 'IAH': 166, 'ICT': 167, 
    'IDA': 168, 'ILM': 169, 'IMT': 170, 'IND': 171, 'INL': 172, 'IPT': 173, 'ISP': 174, 'ITH': 175, 
    'ITO': 176, 'JAC': 177, 'JAN': 178, 'JAX': 179, 'JFK': 180, 'JLN': 181, 'JMS': 182, 'JNU': 183, 
    'KOA': 184, 'KTN': 185, 'LAN': 186, 'LAR': 187, 'LAS': 188, 'LAW': 189, 'LAX': 190, 'LBB': 191, 
    'LBF': 192, 'LBL': 193, 'LCH': 194, 'LEX': 195, 'LFT': 196, 'LGA': 197, 'LGB': 198, 'LIH': 199, 
    'LIT': 200, 'LNK': 201, 'LNY': 202, 'LRD': 203, 'LSE': 204, 'LWB': 205, 'LWS': 206, 'LYH': 207, 
    'MAF': 208, 'MBS': 209, 'MCI': 210, 'MCO': 211, 'MDT': 212, 'MDW': 213, 'MEI': 214, 'MEM': 215, 
    'MFE': 216, 'MFR': 217, 'MGM': 218, 'MHK': 219, 'MHT': 220, 'MIA': 221, 'MKE': 222, 'MKG': 223, 
    'MKK': 224, 'MLB': 225, 'MLI': 226, 'MLU': 227, 'MMH': 228, 'MOB': 229, 'MOT': 230, 'MQT': 231, 
    'MRY': 232, 'MSN': 233, 'MSO': 234, 'MSP': 235, 'MSY': 236, 'MTJ': 237, 'MVY': 238, 'MYR': 239, 
    'OAJ': 240, 'OAK': 241, 'OGD': 242, 'OGG': 243, 'OGS': 244, 'OKC': 245, 'OMA': 246, 'OME': 247, 
    'ONT': 248, 'ORD': 249, 'ORF': 250, 'ORH': 251, 'OTH': 252, 'OTZ': 253, 'OWB': 254, 'PAE': 255, 
    'PAH': 256, 'PBG': 257, 'PBI': 258, 'PDX': 259, 'PGD': 260, 'PGV': 261, 'PHF': 262, 'PHL': 263, 
    'PHX': 264, 'PIA': 265, 'PIB': 266, 'PIE': 267, 'PIH': 268, 'PIR': 269, 'PIT': 270, 'PLN': 271, 
    'PNS': 272, 'PPG': 273, 'PQI': 274, 'PRC': 275, 'PSC': 276, 'PSG': 277, 'PSM': 278, 'PSP': 279, 
    'PUB': 280, 'PUW': 281, 'PVD': 282, 'PVU': 283, 'PWM': 284, 'RAP': 285, 'RDD': 286, 'RDM': 287, 
    'RDU': 288, 'RFD': 289, 'RHI': 290, 'RIC': 291, 'RKS': 292, 'RNO': 293, 'ROA': 294, 'ROC': 295, 
    'ROW': 296, 'RST': 297, 'RSW': 298, 'SAF': 299, 'SAN': 300, 'SAT': 301, 'SAV': 302, 'SBA': 303, 
    'SBN': 304, 'SBP': 305, 'SBY': 306, 'SCC': 307, 'SCE': 308, 'SCK': 309, 'SDF': 310, 'SEA': 311, 
    'SFB': 312, 'SFO': 313, 'SGF': 314, 'SGU': 315, 'SHD': 316, 'SHV': 317, 'SIT': 318, 'SJC': 319, 
    'SJT': 320, 'SJU': 321, 'SLC': 322, 'SLN': 323, 'SMF': 324, 'SMX': 325, 'SNA': 326, 'SPI': 327, 
    'SPN': 328, 'SPS': 329, 'SRQ': 330, 'STC': 331, 'STL': 332, 'STS': 333, 'STT': 334, 'STX': 335, 
    'SUN': 336, 'SUX': 337, 'SWO': 338, 'SYR': 339, 'TLH': 340, 'TOL': 341, 'TPA': 342, 'TRI': 343, 
    'TTN': 344, 'TUL': 345, 'TUS': 346, 'TVC': 347, 'TWF': 348, 'TXK': 349, 'TYR': 350, 'TYS': 351, 
    'UIN': 352, 'USA': 353, 'VEL': 354, 'VLD': 355, 'VPS': 356, 'WRG': 357, 'WYS': 358, 'XNA': 359, 
    'YAK': 360, 'YKM': 361, 'YUM': 362
}

## ====================================================================================================================================== ##

# Function lấy thời tiết
API_KEY = 'd3964c6309d0471c84295728241912'
BASE_URL = 'http://api.weatherapi.com/v1/forecast.json'

def fetch_weather_data(api_key, lat, lon, datetime_obj):
    # Extract the date and time from the datetime object
    date = datetime_obj.strftime('%Y-%m-%d')
    hour = datetime_obj.strftime('%H')

    url = f"{BASE_URL}?key={api_key}&q={lat},{lon}&dt={date}&hour={hour}"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()

        # Check if the forecast data exists for the specified date and time
        if "forecast" in data and "forecastday" in data["forecast"]:
            hour_data = data["forecast"]["forecastday"][0]["hour"][0]

            # Create a dictionary to store the relevant weather information
            weather_info = {
                'Visibility': hour_data.get("vis_km", "N/A"),
                'DryBulbTemperature': hour_data.get("temp_c", "N/A"),
                'Precipitation': hour_data.get("precip_mm", "N/A"),
                'StationPressure': hour_data.get("pressure_in", "N/A"),
                'WindSpeed': hour_data.get("wind_kph", "N/A")
            }
            return weather_info
        else:
            return None
    else:
        return None

def calculate_mean_weather(weather_data_list):
    # Calculate the mean of all weather parameters
    mean_weather = {}

    # For each weather parameter (Visibility, DryBulbTemperature, etc.), calculate the mean
    for key in weather_data_list[0].keys():
        values = [data.get(key) for data in weather_data_list]

        # Filter out "N/A" and calculate the mean if there are valid values
        valid_values = [float(value) for value in values if value != "N/A"]
        if valid_values:
            mean_weather[key] = sum(valid_values) / len(valid_values)
        else:
            mean_weather[key] = "N/A"

    return mean_weather

def extract_mean_weather(api_key, lat, lon, year, month, day, hour, minute):
    # Create a datetime object for the specified date and time
    datetime_obj = datetime(year, month, day, hour, minute)

    # Get the weather data for the current time
    weather_current = fetch_weather_data(api_key, lat, lon, datetime_obj)

    if not weather_current:
        return None

    # Calculate the times for one hour before and one hour after
    time_before = datetime_obj - timedelta(hours=1)
    time_after = datetime_obj + timedelta(hours=1)

    # Get the weather for one hour before
    weather_before = fetch_weather_data(api_key, lat, lon, time_before)

    # Get the weather for one hour after
    weather_after = fetch_weather_data(api_key, lat, lon, time_after)

    # Collect all valid weather data
    weather_data_list = [weather_current]
    if weather_before:
        weather_data_list.append(weather_before)
    if weather_after:
        weather_data_list.append(weather_after)

    # Calculate the mean weather for current, time_before, and time_after
    mean_weather = calculate_mean_weather(weather_data_list)

    return (
        mean_weather['Visibility'],
        mean_weather['DryBulbTemperature'],
        mean_weather['Precipitation'],
        mean_weather['StationPressure'],
        mean_weather['WindSpeed']
    )

## ====================================================================================================================================== ##


def convert_time(start_date, start_time, elapsed_minutes):
    # Kết hợp ngày và giờ khởi hành thành datetime
    start_datetime = datetime.combine(start_date, start_time)
    elapsed_minutes = int(elapsed_minutes)
    # Cộng thêm thời gian bay (phút)
    updated_datetime = start_datetime + timedelta(minutes=elapsed_minutes)
    
    # Tách năm, tháng, ngày, giờ, phút từ thời gian cập nhật
    year = updated_datetime.year
    month = updated_datetime.month
    day = updated_datetime.day
    hour = updated_datetime.hour
    minute = updated_datetime.minute

    # Trả về các giá trị năm, tháng, ngày, giờ, phút
    return year, month, day, hour, minute

## ====================================================================================================================================== ##


# Title and description
st.title("Flight Delay Prediction Interface")
#st.write("Enter the flight details and relevant parameters to predict if the flight will be delayed.")


col1, col2 = st.columns(2)

with col1:
# Inputs for the parameters
#carrier_code = st.number_input("Carrier Code (as a number)", min_value=0, step=1, value=0)
    carrier_code = st.selectbox("Carrier Code",[""] + carrier_names)
    # origin_airport = st.number_input("Origin Airport (as a number)", min_value=0, step=1, value=0)
    origin_airport = st.selectbox('Origin Airport', [""] + origin_airport_options)

    if origin_airport:
        des_options = airport_df[airport_df['origin_airport'] == origin_airport]['destination_airport'].dropna().unique().tolist()
        des_options.sort()
        destination_airport = st.selectbox('Destination Airport', [""] + des_options)
    else:
        destination_airport = st.selectbox('Destination Airport', [""])

    filtered_df  = airport_df[
        (airport_df["origin_airport"] == origin_airport) &
        (airport_df["destination_airport"] == destination_airport)]

    if not filtered_df.empty:
        scheduled_elapsed_time = filtered_df["time_flights"].iloc[0]
    else:
        scheduled_elapsed_time = 0

    scheduled = st.date_input('Departure Date', value = date.today())

    year = scheduled.year
    month = scheduled.month
    day = scheduled.day
    weekday = scheduled.weekday()

    # scheduled_departure_time = st.number_input("Scheduled Departure Time (24-hour format)", min_value=0, max_value=2359, step=1, value=0)

    dep_time = st.time_input("Select flight time (24-hour format) - (HH\:mm)")
    hour = dep_time.hour
    minute = dep_time.minute
    scheduled_departure_time = hour * 60 + minute

    
    scheduled_arrival_time = scheduled_departure_time + scheduled_elapsed_time
    if scheduled_arrival_time > 1440:
        scheduled_arrival_time = scheduled_arrival_time - 1440 
    
## ====================================================================================================================================== ##
## Xử lý thời tiết

    # Thời tiết tại sân bay khởi hành
    HourlyVisibility_x, HourlyDryBulbTemperature_x, HourlyPrecipitation_x, HourlyStationPressure_x, HourlyWindSpeed_x = 0, 0, 0, 0, 0
    if origin_airport: 
        origin_coords = airport_df[airport_df['origin_airport'] == origin_airport][['origin_lat', 'origin_lon']].iloc[0]

        origin_lat = origin_coords['origin_lat']
        origin_lon = origin_coords['origin_lon']
        # year, month, day, hour, minute = 2024, 12, 20, 17, 30

        result_x = extract_mean_weather(API_KEY, origin_lat, origin_lon, year, month, day, hour, minute)
        
        HourlyVisibility_x = result_x[0]
        HourlyDryBulbTemperature_x = result_x[1]
        HourlyPrecipitation_x = result_x[2]
        HourlyStationPressure_x = result_x[3]
        HourlyWindSpeed_x = result_x[4]

    # Thời tiết tại sân bay khởi hành
    HourlyVisibility_y, HourlyDryBulbTemperature_y, HourlyPrecipitation_y, HourlyStationPressure_y, HourlyWindSpeed_y = 0, 0, 0, 0, 0
    if destination_airport:
        destination_coords = airport_df[airport_df['destination_airport'] == destination_airport][['destination_lat', 'destination_lon']]

        destination_lat = destination_coords.iloc[0]['destination_lat']
        destination_lon = destination_coords.iloc[0]['destination_lon']
        year_y, month_y, day_y, hour_y, minute_y = convert_time(scheduled, dep_time, scheduled_elapsed_time)

        result_y = extract_mean_weather(API_KEY, origin_lat, origin_lon, year_y, month_y, day_y, hour_y, minute_y)
        
        HourlyVisibility_y = result_y[0]
        HourlyDryBulbTemperature_y = result_y[1]
        HourlyPrecipitation_y = result_y[2]
        HourlyStationPressure_y = result_y[3]
        HourlyWindSpeed_y = result_y[4]

## ====================================================================================================================================== ##
    
    if weekday == 5 or weekday == 6:
        is_weekend = 1
    else:
        is_weekend = 0
        
        
    holiday = ['1-1', '7-4', '11-11', '11-28', '12-25']
    scheduled_mmdd = f"{scheduled.month}-{scheduled.day}"
    
    if scheduled_mmdd in holiday:
        is_holiday = 1
    else: 
        is_holiday = 0

## ====================================================================================================================================== ##

    flights_per_day = 22693   # median          #st.number_input("Flights Per Day", min_value=0, step=1, value=0)
    delay_rate_last_month = 0.21362 #median     #st.number_input("Delay Rate Last Month", min_value=0.0, max_value=1.0, step=0.01, value=0.0)

with col2:
    if origin_airport and destination_airport:
    # Extract coordinates for origin and destination based on the selected city
        origin_coords = airport_df[airport_df['origin_airport'] == origin_airport][['origin_lat', 'origin_lon']].iloc[0]
        destination_coords = airport_df[airport_df['destination_airport'] == destination_airport][['destination_lat', 'destination_lon']].iloc[0]

        # Coordinates for the map (origin and destination)
        origin_coords = (origin_coords['origin_lat'], origin_coords['origin_lon'])
        destination_coords = (destination_coords['destination_lat'], destination_coords['destination_lon'])

        # Create a folium map centered on the origin
        flight_map = folium.Map(location=origin_coords, zoom_start=5)


        # Add markers for the origin and destination airports
        folium.Marker(location=origin_coords, popup=f"Origin: {origin_airport}").add_to(flight_map)
        folium.Marker(location=destination_coords, popup=f"Destination: {destination_airport}").add_to(flight_map) 
      
        # Draw a polyline for the flight path
        folium.PolyLine([origin_coords, destination_coords], color="red", weight=2.5, opacity=1).add_to(flight_map)

        # Convert the folium map to an HTML string
        map_html = flight_map._repr_html_()

        # Display the map in Streamlit
        st.components.v1.html(map_html, width=600, height=400)
    
#code = carrier_code_dict[carrier_code]

# Create a DataFrame for prediction
input_data = {
    "carrier_code": carrier_code,
    "origin_airport": origin_airport,
    "destination_airport": destination_airport,
    "scheduled_elapsed_time": scheduled_elapsed_time,
    "year": year,
    "month": month,
    "day": day,
    "weekday": weekday,
    "HourlyDryBulbTemperature_x": HourlyDryBulbTemperature_x,
    "HourlyPrecipitation_x": HourlyPrecipitation_x,
    "HourlyStationPressure_x": HourlyStationPressure_x,
    "HourlyVisibility_x": HourlyVisibility_x,
    "HourlyWindSpeed_x": HourlyWindSpeed_x,
    "HourlyDryBulbTemperature_y": HourlyDryBulbTemperature_y,
    "HourlyPrecipitation_y": HourlyPrecipitation_y,
    "HourlyStationPressure_y": HourlyStationPressure_y,
    "HourlyVisibility_y": HourlyVisibility_y,
    "HourlyWindSpeed_y": HourlyWindSpeed_y,
    "is_weekend": is_weekend,
    "scheduled_departure_time": scheduled_departure_time,
    "scheduled_arrival_time": scheduled_arrival_time,
    "is_holiday": is_holiday,
    "flights_per_day": flights_per_day,
    "delay_rate_last_month": delay_rate_last_month,
}

st.write('Data')
input_df = pd.DataFrame([input_data])
st.write(input_df)


if carrier_code in carrier_code_dict:
    input_data['carrier_code'] = carrier_code_dict[carrier_code]
else:
    # Handle the case where the carrier_code is missing or invalid
    input_data['carrier_code'] = None  # Or provide a default value
    

if origin_airport in airport_dict:
    input_data['origin_airport'] = airport_dict[origin_airport]
else:
    # Handle the case where the carrier_code is missing or invalid
    input_data['origin_airport'] = None  # Or provide a default value
    
if destination_airport in airport_dict:
    input_data['destination_airport'] = airport_dict[destination_airport]
else:
    # Handle the case where the carrier_code is missing or invalid
    input_data['destination_airport'] = None  # Or provide a default value
    
    
# filtered_origin_df = airport_df[airport_df['origin_city'] == origin_city]

# # Check if the filtered DataFrame is empty
# if not filtered_origin_df.empty:
#     # Get the origin airport from the filtered DataFrame
#     origin_airport = filtered_origin_df['origin_airport'].iloc[0]
#     input_data['origin_airport'] = airport_dict[origin_airport]
# else:
#     # Handle the case where no matching origin city is found
#     origin_airport = None
#     input_data['origin_airport'] = None


# filtered_df = airport_df[airport_df['destination_city'] == destination_city]

# # Check if the filtered DataFrame is empty
# if not filtered_df.empty:
#     # Get the destination airport from the filtered DataFrame
#     destination_airport = filtered_df['destination_airport'].iloc[0]
#     input_data['destination_airport'] = airport_dict[destination_airport]
# else:
#     # Handle the case where no matching destination city is found
#     destination_airport = None
#     input_data['destination_airport'] = None  # Or provide a default value
    
input_df = pd.DataFrame([input_data])
   
    
# Prediction
if st.button("Predict Delay"):
    try:
        # Preprocess and predict
        prediction = model.predict(input_df)[0]
        if prediction == 1:
            st.success("The flight is predicted to be delayed.")
        else:
            st.success("The flight is predicted to be on time.")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")


