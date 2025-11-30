"""
===============================================================================
File: clean_parking_data.py
Description:
    This script loads raw parking ticket data, cleans and processes it for
    further analysis or machine learning modeling. The steps include:
        - Parsing and standardizing dates and times
        - Extracting numeric and categorical features (hour, month, day of week, etc.)
        - Encoding categorical variables into numeric representations
        - Creating frequency-based features for streets
        - Handling missing or invalid data
        - Saving both a cleaned dataset and a numeric feature dataset for modeling
Usage:
    python clean_parking_data.py
Output:
    - DATA/Final/cleaned_parking_tickets.csv: fully cleaned dataset with readable values
    - DATA/Final/encoded_parking_tickets.csv: numeric encoding suitable for ML models
===============================================================================
"""

import pandas as pd
import os
from datetime import datetime


def clean_parking_data():
    """
    Main function to clean and encode parking ticket data.
    Returns:
        cleaned_data (pd.DataFrame): cleaned and feature-engineered dataframe
    """

    # -------------------------------
    # 1. Define input/output paths
    # -------------------------------
    input_file = "DATA/Initial/Parking_Tickets.csv"
    output_file = "DATA/Final/cleaned_parking_tickets.csv"
    encoded_output_file = "DATA/Final/encoded_parking_tickets.csv"

    # Ensure the output directory exists
    os.makedirs("DATA/Final", exist_ok=True)

    # -------------------------------
    # 2. Load raw data
    # -------------------------------
    print("Loading parking ticket data...")
    df = pd.read_csv(input_file)
    print(f"Original data shape: {df.shape}")

    # Create empty DataFrame to store cleaned features
    cleaned_data = pd.DataFrame()

    # -------------------------------
    # 3. Process and parse dates
    # -------------------------------
    print("Processing dates...")

    # Convert 'DateIssued' to datetime, invalid parsing results in NaT
    df['DateIssued'] = pd.to_datetime(df['DateIssued'], errors='coerce')

    # Remove rows where date parsing failed
    df = df.dropna(subset=['DateIssued'])

    # Filter dataset to include only years 2000-2024
    print("Filtering to years 2000-2024...")
    df = df[(df['DateIssued'].dt.year >= 2000) & (df['DateIssued'].dt.year <= 2024)]
    print(f"Records after year filtering: {len(df)}")

    # Combine DateIssued and TimeIssued into a single datetime column
    df['DateTime'] = pd.to_datetime(
        df['DateIssued'].dt.strftime('%Y-%m-%d') + ' ' + df['TimeIssued'].astype(str),
        errors='coerce'
    )
    # Store as formatted string for cleaned CSV
    cleaned_data['IssuedDate'] = df['DateTime'].dt.strftime('%m/%d/%Y, %I:%M %p')

    # -------------------------------
    # 4. Extract numeric features
    # -------------------------------
    # Hour of day
    cleaned_data['Hour'] = df['DateTime'].dt.hour
    # Month
    cleaned_data['Month'] = df['DateTime'].dt.month
    # Is the day on a weekend? (Saturday=5, Sunday=6)
    cleaned_data['IsWeekend'] = df['DateIssued'].dt.dayofweek.isin([5, 6]).astype(int)

    # -------------------------------
    # 5. Time series features
    # -------------------------------
    print("Adding time series features...")

    # Year
    cleaned_data['Year'] = df['DateIssued'].dt.year
    # Day of week (Monday, Tuesday, etc.)
    cleaned_data['DayOfWeek'] = df['DateIssued'].dt.day_name()

    # Map DayOfWeek to numeric encoding (Mon=0, Sun=6)
    dow_map = {
        'Monday': 0,
        'Tuesday': 1,
        'Wednesday': 2,
        'Thursday': 3,
        'Friday': 4,
        'Saturday': 5,
        'Sunday': 6
    }
    cleaned_data['DayOfWeekEnc'] = cleaned_data['DayOfWeek'].map(dow_map).fillna(-1).astype(int)

    # Quarter of the year
    cleaned_data['Quarter'] = df['DateIssued'].dt.quarter

    # -------------------------------
    # 6. Street name features
    # -------------------------------
    print("Processing street names...")

    # Strip whitespace
    cleaned_data['StreetName'] = df['StreetName'].str.strip()

    # Frequency encoding: how common is each street in dataset
    street_counts = cleaned_data['StreetName'].value_counts(dropna=False)
    street_freq = street_counts / len(cleaned_data)

    # Map street names to their frequency, unknowns get global mean
    street_global_mean = float(street_freq.mean())
    cleaned_data['StreetFreqEnc'] = cleaned_data['StreetName'].map(street_freq).fillna(street_global_mean).astype('float32')

    # -------------------------------
    # 7. Process time features
    # -------------------------------
    print("Processing times...")

    # Convert 'TimeIssued' to string for processing
    time_series = df['TimeIssued'].astype(str)

    # Function to extract hour from various time formats
    def extract_hour(time_str):
        try:
            if ':' in time_str:
                time_obj = datetime.strptime(time_str.strip(), '%H:%M')
                return time_obj.hour
            elif len(time_str.strip()) == 4 and time_str.strip().isdigit():
                return int(time_str.strip()[:2])
            return None
        except:
            return None

    # Map hour to general time period
    def get_time_period(hour):
        if hour is None:
            return 'Unknown'
        elif 6 <= hour < 12:
            return 'Morning'
        elif 12 <= hour < 18:
            return 'Afternoon'
        elif 18 <= hour < 22:
            return 'Evening'
        else:
            return 'Night'

    # Apply extraction functions
    original_hours = time_series.apply(extract_hour)
    cleaned_data['TimePeriod'] = original_hours.apply(get_time_period)

    # Numeric encoding for time periods
    time_period_map = {
        'Night': 0,
        'Morning': 1,
        'Afternoon': 2,
        'Evening': 3,
        'Unknown': -1
    }
    cleaned_data['TimePeriodEnc'] = cleaned_data['TimePeriod'].map(time_period_map).fillna(-1).astype(int)

    # -------------------------------
    # 8. Violation description encoding
    # -------------------------------
    print("Processing violation descriptions...")
    cleaned_data['ViolationDescription'] = df['ViolationDescription']

    # Factorize (ordinal-like encoding) so each unique violation gets a number
    violation_codes, violation_uniques = pd.factorize(cleaned_data['ViolationDescription'], sort=False)
    # Shift by +1 so that unknowns can be coded as 0 later if needed
    cleaned_data['ViolationDescEnc'] = (violation_codes + 1).astype('int32')

    # -------------------------------
    # 9. Drop rows with missing critical data
    # -------------------------------
    print("Removing rows with missing data...")
    initial_count = len(cleaned_data)
    cleaned_data = cleaned_data.dropna(subset=['IssuedDate', 'StreetName', 'ViolationDescription'])
    final_count = len(cleaned_data)
    print(f"Removed {initial_count - final_count} rows with missing data")
    print(f"Final data shape: {cleaned_data.shape}")

    # -------------------------------
    # 10. Save cleaned dataset
    # -------------------------------
    print(f"Saving cleaned data to {output_file}...")
    cleaned_data.to_csv(output_file, index=False, encoding='utf-8')

    # -------------------------------
    # 11. Build numeric feature frame for modeling
    # -------------------------------
    print("Building encoded numeric feature frame...")
    encoded_cols = [
        'Year', 'Quarter', 'Hour', 'Month', 'IsWeekend',
        'DayOfWeekEnc', 'TimePeriodEnc', 'StreetFreqEnc', 'ViolationDescEnc'
    ]
    encoded_df = cleaned_data[encoded_cols].copy()
    # Ensure all columns are numeric
    encoded_df = encoded_df.apply(pd.to_numeric, errors='coerce')

    # Save encoded features
    encoded_df.to_csv(encoded_output_file, index=False, encoding='utf-8')
    print(f"Encoded numeric features saved to: {encoded_output_file}")

    # -------------------------------
    # 12. Preview rarest streets for anomaly analysis
    # -------------------------------
    preview_cols = ['IssuedDate', 'StreetName', 'ViolationDescription', 'StreetFreqEnc']
    print("\nPreview of encoded data (rarest streets first):")
    print(cleaned_data.sort_values('StreetFreqEnc', ascending=True)[preview_cols].head(5))

    print("Data cleaning completed successfully!")
    print(f"Cleaned data saved to: {output_file}")

    # -------------------------------
    # 13. Show sample of cleaned data
    # -------------------------------
    print("\nSample of cleaned data:")
    print(cleaned_data.head()[[
        'IssuedDate','Year','DayOfWeek','DayOfWeekEnc','Quarter','StreetName','StreetFreqEnc',
        'TimePeriod','TimePeriodEnc','ViolationDescription','Hour','Month','IsWeekend'
    ]])

    return cleaned_data


# Entry point: run the cleaning function if script is executed
if __name__ == "__main__":
    clean_parking_data()
