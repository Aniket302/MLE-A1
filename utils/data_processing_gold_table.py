# utils/data_processing_gold_features.py
import os
from datetime import datetime
import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

def process_labels_gold_table(snapshot_date_str, silver_loan_daily_directory, gold_label_store_directory, spark, dpd, mob):
    
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "silver_loan_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_loan_daily_directory + partition_name
    df = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', df.count())

    # get customer at mob
    df = df.filter(col("mob") == mob)

    # get label
    df = df.withColumn("label", F.when(col("dpd") >= dpd, 1).otherwise(0).cast(IntegerType()))
    df = df.withColumn("label_def", F.lit(str(dpd)+'dpd_'+str(mob)+'mob').cast(StringType()))

    # select columns to save
    df = df.select("loan_id", "Customer_ID", "label", "label_def", "snapshot_date")

    # save gold table - IRL connect to database to write
    partition_name = "gold_label_store_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = gold_label_store_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df



def process_feature_store_gold_table(snapshot_date_str, silver_features_dir, gold_feature_store_dir, spark):
    
    # Paths to silver tables for the given snapshot_date
    silver_attributes_path = os.path.join(silver_features_dir, "attributes", f"silver_attributes_{snapshot_date_str.replace('-', '_')}.parquet")
    silver_financials_path = os.path.join(silver_features_dir, "financials", f"silver_financials_{snapshot_date_str.replace('-', '_')}.parquet")
    silver_clickstream_path = os.path.join(silver_features_dir, "clickstream", f"silver_clickstream_{snapshot_date_str.replace('-', '_')}.parquet")

    df_attributes = spark.read.parquet(silver_attributes_path)
    df_financials = spark.read.parquet(silver_financials_path)
    df_clickstream = spark.read.parquet(silver_clickstream_path)

    print(f"Loaded silver_attributes for {snapshot_date_str}, count: {df_attributes.count()}")
    print(f"Loaded silver_financials for {snapshot_date_str}, count: {df_financials.count()}")
    print(f"Loaded silver_clickstream for {snapshot_date_str}, count: {df_clickstream.count()}")

    # Join the tables
    # Use Customer_ID and snapshot_date as join keys.
    # Ensure snapshot_date column name is consistent or aliased if different.
    # For financials and clickstream, snapshot_date is already there. Attributes also has it.
    
    # To avoid ambigous column 'snapshot_date' after join, select it from one df (e.g. attributes)
    # or rename before join. Let's select distinct columns carefully.
    
    join_condition = ["Customer_ID", "snapshot_date"]

    # Start with attributes, then left join financials and clickstream
    # This ensures all customers from attributes are present. If a customer is not in financials or clickstream for that snapshot, their feature values will be null.
    # This seems more robust than inner join.
    
    df_gold = df_attributes.join(df_financials, join_condition, "left") \
                           .join(df_clickstream, join_condition, "left")
    
    # Add a feature_store_snapshot_date for clarity if needed, or rely on the existing snapshot_date
    df_gold = df_gold.withColumn("feature_snapshot_date", F.col("snapshot_date"))

    # Select final columns. Be careful about duplicate column names if any (e.g. Customer_ID, snapshot_date will be unique after join).
    # The join_condition keys will appear once. All other columns from joined tables will be present.

    print(f'Gold feature store for {snapshot_date_str} - joined count: {df_gold.count()}')
    
    # save gold table
    if not os.path.exists(gold_feature_store_dir):
        os.makedirs(gold_feature_store_dir)
        
    partition_name = f"gold_feature_store_{snapshot_date_str.replace('-', '_')}.parquet"
    filepath = os.path.join(gold_feature_store_dir, partition_name)
    df_gold.write.mode("overwrite").parquet(filepath)
    
    print(f'Saved gold feature store to: {filepath}, row count: {df_gold.count()}')
    
    return df_gold