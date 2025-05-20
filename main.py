import os
import glob
import pandas as pd
from datetime import datetime, timedelta
import pyspark

import utils.data_processing_bronze_table
import utils.data_processing_silver_table
import utils.data_processing_gold_table


# Initialize SparkSession
spark = pyspark.sql.SparkSession.builder \
    .appName("dev_feature_store") \
    .master("local[*]") \
    .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
    .getOrCreate()

# Set log level to ERROR to hide warnings
spark.sparkContext.setLogLevel("ERROR")

# set up config
start_date_str = "2023-01-01" 
end_date_str = "2024-12-01" 

# generate list of dates to process
def generate_first_of_month_dates(start_date_str, end_date_str):
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    first_of_month_dates = []
    current_date = datetime(start_date.year, start_date.month, 1)
    while current_date <= end_date:
        first_of_month_dates.append(current_date.strftime("%Y-%m-%d"))
        if current_date.month == 12:
            current_date = datetime(current_date.year + 1, 1, 1)
        else:
            current_date = datetime(current_date.year, current_date.month + 1, 1)
    return first_of_month_dates

dates_str_lst = generate_first_of_month_dates(start_date_str, end_date_str)
print("Processing for dates:", dates_str_lst)

# --- LABEL STORE PIPELINE (Existing) ---
print("\n--- STARTING LABEL STORE PIPELINE ---")
# Create bronze datalake for LMS
bronze_lms_directory = "datamart/bronze/lms/"
if not os.path.exists(bronze_lms_directory):
    os.makedirs(bronze_lms_directory)

# Run bronze backfill for LMS
for date_str in dates_str_lst:
    print(f"\nProcessing LMS Bronze for {date_str}")
    utils.data_processing_bronze_table.process_bronze_label(date_str, bronze_lms_directory, spark)

# Create silver datalake for Loan Daily
silver_loan_daily_directory = "datamart/silver/loan_daily/"
if not os.path.exists(silver_loan_daily_directory):
    os.makedirs(silver_loan_daily_directory)

# Run silver backfill for Loan Daily
for date_str in dates_str_lst:
    print(f"\nProcessing Loan Daily Silver for {date_str}")
    utils.data_processing_silver_table.process_silver_label(date_str, bronze_lms_directory, silver_loan_daily_directory, spark)

# Create gold datalake for Label Store
gold_label_store_directory = "datamart/gold/label_store/"
if not os.path.exists(gold_label_store_directory):
    os.makedirs(gold_label_store_directory)

# Run gold backfill for Label Store
for date_str in dates_str_lst:
    print(f"\nProcessing Label Store Gold for {date_str}")
    utils.data_processing_gold_table.process_labels_gold_table(date_str, silver_loan_daily_directory, gold_label_store_directory, spark, dpd = 30, mob = 6)

print("\n--- LABEL STORE PIPELINE COMPLETED ---")


# --- FEATURE STORE PIPELINE (New) ---
print("\n--- STARTING FEATURE STORE PIPELINE ---")

# Create bronze datalake for Features
bronze_features_directory = "datamart/bronze/features/"
# Subdirectories for attributes, financials, clickstream will be created by functions if not exist

# Run bronze backfill for Features
for date_str in dates_str_lst:
    print(f"\nProcessing Features Bronze for {date_str}")
    utils.data_processing_bronze_table.process_bronze_attributes(date_str, bronze_features_directory, spark)
    utils.data_processing_bronze_table.process_bronze_financials(date_str, bronze_features_directory, spark)
    utils.data_processing_bronze_table.process_bronze_clickstream(date_str, bronze_features_directory, spark)

# Create silver datalake for Features
silver_features_directory = "datamart/silver/features/"
# Subdirectories for attributes, financials, clickstream will be created by functions if not exist

# Run silver backfill for Features
for date_str in dates_str_lst:
    print(f"\nProcessing Features Silver for {date_str}")
    utils.data_processing_silver_table.process_silver_attributes(date_str, bronze_features_directory, silver_features_directory, spark)
    utils.data_processing_silver_table.process_silver_financials(date_str, bronze_features_directory, silver_features_directory, spark)
    utils.data_processing_silver_table.process_silver_clickstream(date_str, bronze_features_directory, silver_features_directory, spark)

# Create gold datalake for Feature Store
gold_feature_store_directory = "datamart/gold/feature_store/"
if not os.path.exists(gold_feature_store_directory):
    os.makedirs(gold_feature_store_directory)

# Run gold backfill for Feature Store
for date_str in dates_str_lst:
    print(f"\nProcessing Feature Store Gold for {date_str}")
    utils.data_processing_gold_table.process_feature_store_gold_table(date_str, silver_features_directory, gold_feature_store_directory, spark)

print("\n--- FEATURE STORE PIPELINE COMPLETED ---")


# --- Display results from Gold Tables ---
print("\n--- GOLD LABEL STORE SAMPLE ---")
label_files_list = [os.path.join(gold_label_store_directory, os.path.basename(f)) for f in glob.glob(os.path.join(gold_label_store_directory, '*.parquet'))]
if label_files_list:
    df_labels_gold = spark.read.option("header", "true").parquet(*label_files_list)
    print("Label Store Gold - row_count:", df_labels_gold.count())
    df_labels_gold.show(5)
else:
    print("No Parquet files found in Gold Label Store directory.")


print("\n--- GOLD FEATURE STORE SAMPLE ---")
feature_files_list = [os.path.join(gold_feature_store_directory, os.path.basename(f)) for f in glob.glob(os.path.join(gold_feature_store_directory, '*.parquet'))]
if feature_files_list:
    df_features_gold = spark.read.option("header", "true").parquet(*feature_files_list)
    print("Feature Store Gold - row_count:", df_features_gold.count())
    df_features_gold.printSchema()
    df_features_gold.show(5)
else:
    print("No Parquet files found in Gold Feature Store directory.")

spark.stop()
print("\n--- SCRIPT COMPLETED ---")