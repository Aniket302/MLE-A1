import os
from datetime import datetime
from pyspark.sql.functions import col

def process_bronze_label(snapshot_date_str, bronze_lms_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to source back end - IRL connect to back end source system
    csv_file_path = "data/lms_loan_daily.csv"

    # load data - IRL ingest from back end source system
    df = spark.read.csv(csv_file_path, header=True, inferSchema=True).filter(col('snapshot_date') == snapshot_date)
    print(snapshot_date_str + 'row count:', df.count())

    # save bronze table to datamart - IRL connect to database to write
    partition_name = "bronze_loan_daily_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_lms_directory + partition_name
    df.toPandas().to_csv(filepath, index=False)
    print('saved to:', filepath)

    return df


# Common function to process a generic feature CSV to bronze
def process_feature_to_bronze(snapshot_date_str, bronze_base_dir, feature_name, raw_csv_path, spark):
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    bronze_feature_directory = os.path.join(bronze_base_dir, feature_name)
    if not os.path.exists(bronze_feature_directory):
        os.makedirs(bronze_feature_directory)

    df = spark.read.csv(raw_csv_path, header=True, inferSchema=True)

    # Filter by snapshot_date
    df_filtered = df.filter(col('snapshot_date') == snapshot_date)
    
    print(f"{feature_name} - {snapshot_date_str} raw row count: {df.count()}, filtered row count: {df_filtered.count()}")

    partition_name = f"bronze_{feature_name}_{snapshot_date_str.replace('-', '_')}.csv"
    filepath = os.path.join(bronze_feature_directory, partition_name)

    df_filtered.toPandas().to_csv(filepath, index=False)
    print(f'Saved {feature_name} bronze to: {filepath}')
    return df_filtered


def process_bronze_attributes(snapshot_date_str, bronze_features_directory, spark):
    raw_csv_path = "data/features_attributes.csv"
    return process_feature_to_bronze(snapshot_date_str, bronze_features_directory, "attributes", raw_csv_path, spark)

def process_bronze_financials(snapshot_date_str, bronze_features_directory, spark):
    raw_csv_path = "data/features_financials.csv"
    return process_feature_to_bronze(snapshot_date_str, bronze_features_directory, "financials", raw_csv_path, spark)

def process_bronze_clickstream(snapshot_date_str, bronze_features_directory, spark):
    raw_csv_path = "data/feature_clickstream.csv"
    return process_feature_to_bronze(snapshot_date_str, bronze_features_directory, "clickstream", raw_csv_path, spark)