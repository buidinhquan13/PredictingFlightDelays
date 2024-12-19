import streamlit as st
import pandas as pd
import joblib

#st.title("🚗 FLIGHT DELAY PREDICTION 🚗")


airport_df = pd.read_csv('airport_coordinates.csv')
origin_airport_options = airport_df['origin_city'].dropna().unique().tolist()
origin_airport_options.sort()

destination_airport_options = airport_df['destination_city'].dropna().unique().tolist()
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

# Title and description
st.title("Flight Delay Prediction Interface")
#st.write("Enter the flight details and relevant parameters to predict if the flight will be delayed.")

# Inputs for the parameters
#carrier_code = st.number_input("Carrier Code (as a number)", min_value=0, step=1, value=0)
carrier_code = st.selectbox("Carrier Code",[""] + carrier_names)
# origin_airport = st.number_input("Origin Airport (as a number)", min_value=0, step=1, value=0)
origin_city = st.selectbox('Origin City', [""] + origin_airport_options)

if origin_city:
    des_options = airport_df[airport_df['origin_city'] == origin_city]['destination_city'].dropna().unique().tolist()
    des_options.sort()
    destination_city = st.selectbox('Destination City', [""] + des_options)
else:
    destination_city = st.selectbox('Destination City', [""])


# destination_airport = st.selectbox('Destination Airport', [""] + des_options)

    
# destination_airport = st.number_input("Destination Airport (as a number)", min_value=0, step=1, value=0)


filtered_df  = airport_df[
    (airport_df["origin_city"] == origin_city) &
    (airport_df["destination_city"] == destination_city)]

if not filtered_df.empty:
    scheduled_elapsed_time = filtered_df["time_flights"].iloc[0]
else:
    scheduled_elapsed_time = 0

year = st.number_input("Year", min_value=2000, step=1, value=2024)
month = st.number_input("Month", min_value=1, max_value=12, step=1, value=1)
day = st.number_input("Day", min_value=1, max_value=31, step=1, value=1)
weekday = st.number_input("Weekday (0=Monday, 6=Sunday)", min_value=0, max_value=6, step=1, value=0)
HourlyDryBulbTemperature_x = st.number_input("Hourly Dry Bulb Temperature (Origin)", value=0.0)
HourlyPrecipitation_x = st.number_input("Hourly Precipitation (Origin)", value=0.0)
HourlyStationPressure_x = st.number_input("Hourly Station Pressure (Origin)", value=0.0)
HourlyVisibility_x = st.number_input("Hourly Visibility (Origin)", value=0.0)
HourlyWindSpeed_x = st.number_input("Hourly Wind Speed (Origin)", value=0.0)
HourlyDryBulbTemperature_y = st.number_input("Hourly Dry Bulb Temperature (Destination)", value=0.0)
HourlyPrecipitation_y = st.number_input("Hourly Precipitation (Destination)", value=0.0)
HourlyStationPressure_y = st.number_input("Hourly Station Pressure (Destination)", value=0.0)
HourlyVisibility_y = st.number_input("Hourly Visibility (Destination)", value=0.0)
HourlyWindSpeed_y = st.number_input("Hourly Wind Speed (Destination)", value=0.0)
is_weekend = st.selectbox("Is Weekend", options=[0, 1])
scheduled_departure_time = st.number_input("Scheduled Departure Time (24-hour format)", min_value=0, max_value=2359, step=1, value=0)
scheduled_arrival_time = st.number_input("Scheduled Arrival Time (24-hour format)", min_value=0, max_value=2359, step=1, value=0)
is_holiday = st.selectbox("Is Holiday", options=[0, 1])
flights_per_day = st.number_input("Flights Per Day", min_value=0, step=1, value=0)
delay_rate_last_month = st.number_input("Delay Rate Last Month", min_value=0.0, max_value=1.0, step=0.01, value=0.0)




#code = carrier_code_dict[carrier_code]

# Create a DataFrame for prediction
input_data = {
    "carrier_code": carrier_code,
    "origin_airport": origin_city,
    "destination_airport": destination_city,
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

input_df = pd.DataFrame([input_data])

# Display the input parameters as a DataFrame
if st.button("Show Input Summary"):
    st.write(input_df)
    
# Xử lý data
# input_data['carrier_code'] = carrier_code_dict[carrier_code]

if carrier_code in carrier_code_dict:
    input_data['carrier_code'] = carrier_code_dict[carrier_code]
else:
    # Handle the case where the carrier_code is missing or invalid
    input_data['carrier_code'] = None  # Or provide a default value
    
    
# origin_airport = airport_df[airport_df['origin_city'] == origin_city]['origin_airport'].iloc[0]
# input_data['origin_airport'] = airport_dict[origin_airport]

filtered_origin_df = airport_df[airport_df['origin_city'] == origin_city]

# Check if the filtered DataFrame is empty
if not filtered_origin_df.empty:
    # Get the origin airport from the filtered DataFrame
    origin_airport = filtered_origin_df['origin_airport'].iloc[0]
    input_data['origin_airport'] = airport_dict[origin_airport]
else:
    # Handle the case where no matching origin city is found
    origin_airport = None
    input_data['origin_airport'] = None

# destination_airport = airport_df[airport_df['destination_city'] == destination_city]['destination_airport'].iloc[0]
# input_data['destination_airport'] = airport_dict[destination_airport]

filtered_df = airport_df[airport_df['destination_city'] == destination_city]

# Check if the filtered DataFrame is empty
if not filtered_df.empty:
    # Get the destination airport from the filtered DataFrame
    destination_airport = filtered_df['destination_airport'].iloc[0]
    input_data['destination_airport'] = airport_dict[destination_airport]
else:
    # Handle the case where no matching destination city is found
    destination_airport = None
    input_data['destination_airport'] = None  # Or provide a default value
    
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


