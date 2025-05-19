# utils/data_processing_silver_features.py
import os
import re
from datetime import datetime
import pyspark.sql.functions as F
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType, DoubleType
from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

def process_silver_label(snapshot_date_str, bronze_lms_directory, silver_loan_daily_directory, spark):

    # connect to bronze table
    partition_name = "bronze_loan_daily_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_lms_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    # clean data: enforce schema / data type
    # Dictionary specifying columns and their desired datatypes
    column_type_map = {
        "loan_id": StringType(),
        "Customer_ID": StringType(),
        "loan_start_date": DateType(),
        "tenure": IntegerType(),
        "installment_num": IntegerType(),
        "loan_amt": FloatType(),
        "due_amt": FloatType(),
        "paid_amt": FloatType(),
        "overdue_amt": FloatType(),
        "balance": FloatType(),
        "snapshot_date": DateType(),
    }

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # augment data: add month on book
    df = df.withColumn("mob", col("installment_num").cast(IntegerType()))

    # augment data: add days past due
    df = df.withColumn("installments_missed", F.ceil(col("overdue_amt") / col("due_amt")).cast(IntegerType())).fillna(0)
    df = df.withColumn("first_missed_date", F.when(col("installments_missed") > 0, F.add_months(col("snapshot_date"), -1 * col("installments_missed"))).cast(DateType()))
    df = df.withColumn("dpd", F.when(col("overdue_amt") > 0.0, F.datediff(col("snapshot_date"), col("first_missed_date"))).otherwise(0).cast(IntegerType()))

    # save silver table - IRL connect to database to write
    partition_name = "silver_loan_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_loan_daily_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)
    
    return df



def _parse_credit_history_age_func(age_str):
    if age_str is None:
        return None
    try:
        years = 0
        months = 0
        year_match = re.search(r"(\d+)\s*Years?", str(age_str))
        month_match = re.search(r"(\d+)\s*Months?", str(age_str))
        if year_match:
            years = int(year_match.group(1))
        if month_match:
            months = int(month_match.group(1))
        return years * 12 + months
    except Exception:
        return None

parse_credit_history_age_udf = F.udf(_parse_credit_history_age_func, IntegerType())


def process_silver_attributes(snapshot_date_str, bronze_features_dir, silver_features_dir, spark):
    bronze_file_path = os.path.join(bronze_features_dir, "attributes", f"bronze_attributes_{snapshot_date_str.replace('-', '_')}.csv")
    silver_attributes_directory = os.path.join(silver_features_dir, "attributes")
    if not os.path.exists(silver_attributes_directory):
        os.makedirs(silver_attributes_directory)

    df = spark.read.csv(bronze_file_path, header=True, inferSchema=True)
    print(f'Loaded attributes from: {bronze_file_path}, row count: {df.count()}')

    # -------- data-cleaning BEFORE casting --------
    df = (
        df
        # 1. Age: strip underscores, convert to int, drop impossible values
        .withColumn(
            "Age",
            F.regexp_replace("Age", "_", "")          # get rid of underscores
             .cast(IntegerType())                      # cast to int
        )
        .filter((F.col("Age") >= 0) & (F.col("Age") <= 110))  # keep realistic ages
        
        # 2. Occupation: map "_______" ➜ "Others" (case-insensitive, trim just in case)
        .withColumn(
            "Occupation",
            F.when(F.trim(F.lower("Occupation")) == "_______", "Others")
             .otherwise(F.col("Occupation"))
        )
    )

    # Enforce schema / data type
    df = df.withColumn("Customer_ID", F.col("Customer_ID").cast(StringType())) \
           .withColumn("Age", F.col("Age").cast(IntegerType())) \
           .withColumn("Occupation", F.col("Occupation").cast(StringType())) \
           .withColumn("snapshot_date", F.to_date(F.col("snapshot_date"), "yyyy-MM-dd")) # Ensure date type

    # Name and SSN not relevant features
    df = df.select("Customer_ID", "Age", "Occupation", "snapshot_date") # Select and order

    # save silver table
    partition_name = f"silver_attributes_{snapshot_date_str.replace('-', '_')}.parquet"
    filepath = os.path.join(silver_attributes_directory, partition_name)
    df.write.mode("overwrite").parquet(filepath)
    print(f'Saved silver attributes to: {filepath}, row count: {df.count()}')
    return df


# 0 Customer_ID               12500 non-null  object 
#  1   Annual_Income             12500 non-null  object -> remove underscore
#  2   Monthly_Inhand_Salary     12500 non-null  float64 
#  3   Num_Bank_Accounts         12500 non-null  int64 -> remove high bank accounts, remove negative 
#  4   Num_Credit_Card           12500 non-null  int64  -> remove high credit cards
#  5   Interest_Rate             12500 non-null  int64  -> remove interest rate greater than 100
#  6   Num_of_Loan               12500 non-null  object -> remove underscore, remove negative, remove high num of loan
#  7   Type_of_Loan              11074 non-null  object -> 
#  8   Delay_from_due_date       12500 non-null  int64  
#  9   Num_of_Delayed_Payment    12500 non-null  object 
#  10  Changed_Credit_Limit      12500 non-null  object 
#  11  Num_Credit_Inquiries      12500 non-null  float64
#  12  Credit_Mix                12500 non-null  object 
#  13  Outstanding_Debt          12500 non-null  object 
#  14  Credit_Utilization_Ratio  12500 non-null  float64
#  15  Credit_History_Age        12500 non-null  object 
#  16  Payment_of_Min_Amount     12500 non-null  object 
#  17  Total_EMI_per_month       12500 non-null  float64
#  18  Amount_invested_monthly   12500 non-null  object 
#  19  Payment_Behaviour         12500 non-null  object 
#  20  Monthly_Balance           12500 non-null  object 
#  21  snapshot_date             12500 non-null  object 


def process_silver_financials(snapshot_date_str, bronze_features_dir, silver_features_dir, spark):
    bronze_file_path = os.path.join(bronze_features_dir, "financials", f"bronze_financials_{snapshot_date_str.replace('-', '_')}.csv")
    silver_financials_directory = os.path.join(silver_features_dir, "financials")
    if not os.path.exists(silver_financials_directory):
        os.makedirs(silver_financials_directory)

    df = spark.read.csv(bronze_file_path, header=True, inferSchema=True) # Keep inferSchema for now, cast explicitly
    print(f'Loaded financials from: {bronze_file_path}, row count: {df.count()}')

    # Clean and cast columns
    df = df.withColumn("snapshot_date", F.to_date(F.col("snapshot_date"), "yyyy-MM-dd")) \
           .withColumn("Annual_Income", F.regexp_replace(F.col("Annual_Income"), "[^0-9.]", "").cast(FloatType())) \
           .withColumn("Monthly_Inhand_Salary", F.col("Monthly_Inhand_Salary").cast(FloatType())) \
           .withColumn("Num_Bank_Accounts", F.col("Num_Bank_Accounts").cast(IntegerType())) \
           .withColumn("Num_Credit_Card", F.col("Num_Credit_Card").cast(IntegerType())) \
           .withColumn("Interest_Rate", F.col("Interest_Rate").cast(IntegerType())) \
           .withColumn("Num_of_Loan", F.regexp_replace(F.col("Num_of_Loan"), "[^0-9]", "").cast(IntegerType())) \
           .withColumn("Delay_from_due_date", F.col("Delay_from_due_date").cast(IntegerType())) \
           .withColumn("Num_of_Delayed_Payment", F.regexp_replace(F.col("Num_of_Delayed_Payment"), "[^0-9]", "").cast(IntegerType())) \
           .withColumn("Changed_Credit_Limit", F.regexp_replace(F.col("Changed_Credit_Limit"), "_", "").cast(FloatType())) \
           .withColumn("Num_Credit_Inquiries", F.col("Num_Credit_Inquiries").cast(IntegerType())) \
           .withColumn("Credit_Mix", F.when(F.col("Credit_Mix") == "_", "Unknown").otherwise(F.col("Credit_Mix")).cast(StringType())) \
           .withColumn("Outstanding_Debt", F.regexp_replace(F.col("Outstanding_Debt"), "[^0-9.]", "").cast(FloatType())) \
           .withColumn("Credit_Utilization_Ratio", F.col("Credit_Utilization_Ratio").cast(FloatType())) \
           .withColumn("Credit_History_Age_Months", parse_credit_history_age_udf(F.col("Credit_History_Age"))) \
           .withColumn("Payment_of_Min_Amount", 
                       F.when(F.col("Payment_of_Min_Amount") == "Yes", 1)
                        .when(F.col("Payment_of_Min_Amount") == "No", 0)
                        .otherwise(None).cast(IntegerType())) \
           .withColumn("Total_EMI_per_month", F.col("Total_EMI_per_month").cast(FloatType())) \
           .withColumn("Amount_invested_monthly", F.regexp_replace(F.col("Amount_invested_monthly"), "[^0-9.]", "").cast(FloatType())) \
           .withColumn("Monthly_Balance", F.regexp_replace(F.col("Monthly_Balance"), "[^0-9.]", "").cast(FloatType()))

    # Select relevant columns, drop original complex string columns if parsed
    df = df.select(
        "Customer_ID", "snapshot_date", "Annual_Income", "Monthly_Inhand_Salary",
        "Num_Bank_Accounts", "Num_Credit_Card", "Interest_Rate", "Num_of_Loan", "Type_of_Loan",
        "Delay_from_due_date", "Num_of_Delayed_Payment", "Changed_Credit_Limit",
        "Num_Credit_Inquiries", "Credit_Mix", "Outstanding_Debt", "Credit_Utilization_Ratio",
        "Credit_History_Age_Months", "Payment_of_Min_Amount", "Total_EMI_per_month",
        "Amount_invested_monthly", "Payment_Behaviour", "Monthly_Balance"
    )
    
    # save silver table
    partition_name = f"silver_financials_{snapshot_date_str.replace('-', '_')}.parquet"
    filepath = os.path.join(silver_financials_directory, partition_name)
    df.write.mode("overwrite").parquet(filepath)
    print(f'Saved silver financials to: {filepath}')
    return df




def process_silver_clickstream(snapshot_date_str, bronze_features_dir, silver_features_dir, spark):
    bronze_file_path = os.path.join(bronze_features_dir, "clickstream", f"bronze_clickstream_{snapshot_date_str.replace('-', '_')}.csv")
    silver_clickstream_directory = os.path.join(silver_features_dir, "clickstream")
    if not os.path.exists(silver_clickstream_directory):
        os.makedirs(silver_clickstream_directory)

    df = spark.read.csv(bronze_file_path, header=True, inferSchema=True)
    print(f'Loaded clickstream from: {bronze_file_path}, row count: {df.count()}')

    # Ensure snapshot_date is DateType
    df = df.withColumn("snapshot_date", F.to_date(F.col("snapshot_date"), "yyyy-MM-dd"))

    # Cast all fe_X columns to IntegerType just to be sure, though inferSchema might get them right
    for i in range(1, 21):
        col_name = f"fe_{i}"
        if col_name in df.columns:
            df = df.withColumn(col_name, F.col(col_name).cast(IntegerType()))
    
    df = df.withColumn("Customer_ID", F.col("Customer_ID").cast(StringType()))

    # save silver table
    partition_name = f"silver_clickstream_{snapshot_date_str.replace('-', '_')}.parquet"
    filepath = os.path.join(silver_clickstream_directory, partition_name)
    df.write.mode("overwrite").parquet(filepath)
    print(f'Saved silver clickstream to: {filepath}')
    return df