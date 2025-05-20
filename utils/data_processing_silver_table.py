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



def process_silver_attributes(snapshot_date_str, bronze_features_dir, silver_features_dir, spark):
    bronze_file_path = os.path.join(bronze_features_dir, "attributes", f"bronze_attributes_{snapshot_date_str.replace('-', '_')}.csv")
    silver_attributes_directory = os.path.join(silver_features_dir, "attributes")
    if not os.path.exists(silver_attributes_directory):
        os.makedirs(silver_attributes_directory)

    df = spark.read.csv(bronze_file_path, header=True, inferSchema=True)
    print(f'Loaded attributes from: {bronze_file_path}, row count: {df.count()}')

    # -------------------------------
    # Data Cleaning
    # -------------------------------
    df = (
        df
        # 0. Customer_ID: ensure starts with 'CUS_'
        .withColumn("Customer_ID", F.when(F.col("Customer_ID").startswith("CUS_"), F.col("Customer_ID"))
                                      .otherwise(F.concat(F.lit("CUS_"), F.col("Customer_ID"))))
        
        # 1. Age: strip underscores, convert to int, drop values not in range 18 - 100
        .withColumn(
            "Age",
            F.regexp_replace("Age", "_", "")          
             .cast(IntegerType())                      
        )
        .filter((F.col("Age") >= 18) & (F.col("Age") <= 100))  
        
        # 2. Occupation: map "_______" ➜ "Others" 
        .withColumn(
            "Occupation",
            F.when(F.trim(F.lower("Occupation")) == "_______", "Others")
             .otherwise(F.col("Occupation"))
        )
    )


    # -------------------------------
    # Data Type Casting
    # -------------------------------
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


# 0 Customer_ID               12500 non-null  object -> Make sure every value begins with 'CUS_' (string)
#  1   Annual_Income             12500 non-null  object -> remove underscore (float)
#  2   Monthly_Inhand_Salary     12500 non-null  float64 -> remove negatives (float)
#  3   Num_Bank_Accounts         12500 non-null  int64 -> remove remove negative (int)
#  4   Num_Credit_Card           12500 non-null  int64  -> remove negative credit cards (int)
#  5   Interest_Rate             12500 non-null  int64  -> Keep between 1 and 100 (float)
#  6   Num_of_Loan               12500 non-null  object -> remove underscore, remove negative (float)
#  7   Type_of_Loan              11074 non-null  object -> fill na with 'Unknown' (string) (categorical)
#  8   Delay_from_due_date       12500 non-null  int64  -> keep between 0 to max value (remove negatives) (int)
#  9   Num_of_Delayed_Payment    12500 non-null  object -> remove _, remove negatives (int), replace missing values with 0
#  10  Changed_Credit_Limit      12500 non-null  object -> remove underscore (float), remove negative values
#  11  Num_Credit_Inquiries      12500 non-null  float64 -> remove negatives (float)
#  12  Credit_Mix                12500 non-null  object -> replace '_' with 'Unknown' (string) (categorical)
#  13  Outstanding_Debt          12500 non-null  object -> remove underscore (float)
#  14  Credit_Utilization_Ratio  12500 non-null  float64 -> keep between 0 and 100.0 (float)
#  15  Credit_History_Age        12500 non-null  object -> its in the format of "X Years Y Months". Replace NA with 0 and just keep X years (dfMerged['Credit_History_Age'] = dfMerged['Credit_History_Age'].apply(lambda x: '0' if x == "NA" else str(x).split()[0]).astype(int)) (int)
#  16  Payment_of_Min_Amount     12500 non-null  object -> (string) (categorical)
#  17  Total_EMI_per_month       12500 non-null  float64 -> (float)
#  18  Amount_invested_monthly   12500 non-null  object -> replace na with 0, remove underscores, there is a value like '__10000__' (float)
#  19  Payment_Behaviour         12500 non-null  object -> replace('!@9#%8', 'Unknown'), (string) (categorical)
#  20  Monthly_Balance           12500 non-null  object -> remove underscores, remove negative values (float)
#  21  snapshot_date             12500 non-null  object -> (date)


def process_silver_financials(snapshot_date_str, bronze_features_dir, silver_features_dir, spark):
    bronze_file_path = os.path.join(bronze_features_dir, "financials", f"bronze_financials_{snapshot_date_str.replace('-', '_')}.csv")
    silver_financials_directory = os.path.join(silver_features_dir, "financials")
    if not os.path.exists(silver_financials_directory):
        os.makedirs(silver_financials_directory)

    df = spark.read.csv(bronze_file_path, header=True, inferSchema=True) # Keep inferSchema for now, cast explicitly
    print(f'Loaded financials from: {bronze_file_path}, row count: {df.count()}')

    # -------------------------------
    # Data Cleaning
    # -------------------------------

    # df = (
    #     df
    #     # 0. Customer_ID: ensure starts with 'CUS_'
    #     .withColumn("Customer_ID", F.when(F.col("Customer_ID").startswith("CUS_"), F.col("Customer_ID"))
    #                                   .otherwise(F.concat(F.lit("CUS_"), F.col("Customer_ID"))))

    #     # 1. Annual_Income: remove underscores
    #     .withColumn("Annual_Income", F.regexp_replace("Annual_Income", "_", ""))

    #     # 2. Monthly_Inhand_Salary: remove negatives
    #     .withColumn("Monthly_Inhand_Salary", F.when(F.col("Monthly_Inhand_Salary") < 0, None)
    #                                            .otherwise(F.col("Monthly_Inhand_Salary")))

    #     # 3. Num_Bank_Accounts: remove negatives
    #     .withColumn("Num_Bank_Accounts", F.when(F.col("Num_Bank_Accounts") < 0, None)
    #                                        .otherwise(F.col("Num_Bank_Accounts")))

    #     # 4. Num_Credit_Card: remove negatives
    #     .withColumn("Num_Credit_Card", F.when(F.col("Num_Credit_Card") < 0, None)
    #                                      .otherwise(F.col("Num_Credit_Card")))

    #     # 5. Interest_Rate: keep between 1 and 100
    #     .withColumn("Interest_Rate", F.when((F.col("Interest_Rate") >= 1) & (F.col("Interest_Rate") <= 100), F.col("Interest_Rate"))
    #                                    .otherwise(None))

    #     # 6. Num_of_Loan: remove underscores, remove negatives
    #     .withColumn("Num_of_Loan", F.regexp_replace("Num_of_Loan", "_", "").cast(FloatType()))
    #     .withColumn("Num_of_Loan", F.when(F.col("Num_of_Loan") < 0, None)
    #                                  .otherwise(F.col("Num_of_Loan")))

    #     # 7. Type_of_Loan: fill NA with 'Unknown'
    #     .withColumn("Type_of_Loan", F.when(F.col("Type_of_Loan").isNull(), "Unknown")
    #                                   .otherwise(F.col("Type_of_Loan")))

    #     # 8. Delay_from_due_date: remove negatives
    #     .withColumn("Delay_from_due_date", F.when(F.col("Delay_from_due_date") < 0, None)
    #                                          .otherwise(F.col("Delay_from_due_date")))

    #     # 9. Num_of_Delayed_Payment: remove_, remove negatives
    #     .withColumn(
    #             "Num_of_Delayed_Payment",
    #             F.regexp_replace("Num_of_Delayed_Payment", "_", "")
    #         ).withColumn(
    #             "Num_of_Delayed_Payment",
    #             F.when(F.col("Num_of_Delayed_Payment") < 0, 0).otherwise(F.col("Num_of_Delayed_Payment"))
    #         )

    #     # 10. Changed_Credit_Limit: remove underscores
    #     .withColumn("Changed_Credit_Limit", F.regexp_replace("Changed_Credit_Limit", "_", ""))

    #     # 11. Num_Credit_Inquiries: remove negatives
    #     .withColumn("Num_Credit_Inquiries", F.when(F.col("Num_Credit_Inquiries") < 0, None)
    #                                           .otherwise(F.col("Num_Credit_Inquiries")))

    #     # 12. Credit_Mix: replace '_' with 'Unknown'
    #     .withColumn("Credit_Mix", F.when(F.col("Credit_Mix") == "_", "Unknown")
    #                                 .otherwise(F.col("Credit_Mix")))

    #     # 13. Outstanding_Debt: remove underscores
    #     .withColumn("Outstanding_Debt", F.regexp_replace("Outstanding_Debt", "_", ""))

    #     # 14. Credit_Utilization_Ratio: keep between 0 and 100.0
    #     .withColumn("Credit_Utilization_Ratio", F.when((F.col("Credit_Utilization_Ratio") >= 0) & (F.col("Credit_Utilization_Ratio") <= 100.0),
    #                                                     F.col("Credit_Utilization_Ratio"))
    #                                              .otherwise(None))

    #     # 15. Credit_History_Age: "X Years Y Months" → extract X, replace NA with 0
    #     .withColumn("Credit_History_Age", F.when(F.col("Credit_History_Age").isNull(), 0)
    #                                        .otherwise(F.split(F.col("Credit_History_Age"), " ").getItem(0)))

    #     # 18. Amount_invested_monthly: replace NA with 0, remove underscores
    #     .withColumn("Amount_invested_monthly", F.when(F.col("Amount_invested_monthly").isNull(), 0)
    #                                              .otherwise(F.col("Amount_invested_monthly")))
    #     .withColumn("Amount_invested_monthly", F.regexp_replace("Amount_invested_monthly", "_", ""))

    #     # 19. Payment_Behaviour: replace '!@9#%8' with 'Unknown'
    #     .withColumn("Payment_Behaviour", F.when(F.col("Payment_Behaviour") == "!@9#%8", "Unknown")
    #                                        .otherwise(F.col("Payment_Behaviour")))

    #     # 20. Monthly_Balance: remove underscores, remove negatives
    #     .withColumn("Monthly_Balance", F.regexp_replace("Monthly_Balance", "_", "").cast(FloatType()))
    #     .withColumn("Monthly_Balance", F.when(F.col("Monthly_Balance") < 0, None)
    #                                      .otherwise(F.col("Monthly_Balance")))
    # )

    df = (
        df
        # 0. Customer_ID: ensure starts with 'CUS_'
        .withColumn("Customer_ID", F.when(F.col("Customer_ID").startswith("CUS_"), F.col("Customer_ID"))
                                    .otherwise(F.concat(F.lit("CUS_"), F.col("Customer_ID"))))

        # 1. Annual_Income: remove underscores
        .withColumn("Annual_Income", F.regexp_replace("Annual_Income", "_", ""))

        # 2. Monthly_Inhand_Salary: will be filtered later
        # 3. Num_Bank_Accounts: will be filtered later
        # 4. Num_Credit_Card: will be filtered later

        # 5. Interest_Rate: keep between 1 and 100 (filter later)
        
        # 6. Num_of_Loan: remove underscores, filter later
        .withColumn("Num_of_Loan", F.regexp_replace("Num_of_Loan", "_", "").cast(FloatType()))

        # 7. Type_of_Loan: fill NA with 'Unknown'
        .withColumn("Type_of_Loan", F.when(F.col("Type_of_Loan").isNull(), "Unknown")
                                    .otherwise(F.col("Type_of_Loan")))

        # 8. Delay_from_due_date: will be filtered later

        # 9. Num_of_Delayed_Payment: remove underscore, cast to float for now
        .withColumn("Num_of_Delayed_Payment", F.regexp_replace("Num_of_Delayed_Payment", "_", "").cast(FloatType()))

        # 10. Changed_Credit_Limit: remove underscores, greater than 0
        .withColumn("Changed_Credit_Limit", F.regexp_replace("Changed_Credit_Limit", "_", "").cast(FloatType()))

        # 11. Num_Credit_Inquiries: will be filtered later

        # 12. Credit_Mix: replace '_' with 'Unknown'
        .withColumn("Credit_Mix", F.when(F.col("Credit_Mix") == "_", "Unknown")
                                    .otherwise(F.col("Credit_Mix")))

        # 13. Outstanding_Debt: remove underscores
        .withColumn("Outstanding_Debt", F.regexp_replace("Outstanding_Debt", "_", ""))

        # 14. Credit_Utilization_Ratio: filter later

        # 15. Credit_History_Age: "X Years Y Months" → extract X, replace NA with 0
        .withColumn("Credit_History_Age", F.when(F.col("Credit_History_Age").isNull(), 0)
                                        .otherwise(F.split(F.col("Credit_History_Age"), " ").getItem(0)))

        # 18. Amount_invested_monthly: replace NA with 0, remove underscores
        .withColumn("Amount_invested_monthly", F.when(F.col("Amount_invested_monthly").isNull(), 0)
                                                .otherwise(F.col("Amount_invested_monthly")))
        .withColumn("Amount_invested_monthly", F.regexp_replace("Amount_invested_monthly", "_", ""))

        # 19. Payment_Behaviour: replace '!@9#%8' with 'Unknown'
        .withColumn("Payment_Behaviour", F.when(F.col("Payment_Behaviour") == "!@9#%8", "Unknown")
                                        .otherwise(F.col("Payment_Behaviour")))

        # 20. Monthly_Balance: remove underscores, cast to float
        .withColumn("Monthly_Balance", F.regexp_replace("Monthly_Balance", "_", "").cast(FloatType()))
    )

    # Apply .filter() to drop invalid customer data
    df = (
        df
        .filter(F.col("Monthly_Inhand_Salary") >= 0)
        .filter(F.col("Num_Bank_Accounts") >= 0)
        .filter(F.col("Num_Credit_Card") >= 0)
        .filter((F.col("Interest_Rate") >= 1) & (F.col("Interest_Rate") <= 100))
        .filter(F.col("Num_of_Loan") >= 0)
        .filter(F.col("Delay_from_due_date") >= 0)
        .filter(F.col("Changed_Credit_Limit") > 0)
        .filter(F.col("Num_of_Delayed_Payment") >= 0)
        .filter(F.col("Num_Credit_Inquiries") >= 0)
        .filter((F.col("Credit_Utilization_Ratio") >= 0) & (F.col("Credit_Utilization_Ratio") <= 100))
        .filter(F.col("Monthly_Balance") >= 0)   
    )

    # -------------------------------
    # Data Type Casting
    # -------------------------------

    df = (
        df
        .withColumn("Customer_ID", F.col("Customer_ID").cast(StringType()))
        .withColumn("Annual_Income", F.col("Annual_Income").cast(FloatType()))
        .withColumn("Monthly_Inhand_Salary", F.col("Monthly_Inhand_Salary").cast(FloatType()))
        .withColumn("Num_Bank_Accounts", F.col("Num_Bank_Accounts").cast(IntegerType()))
        .withColumn("Num_Credit_Card", F.col("Num_Credit_Card").cast(IntegerType()))
        .withColumn("Interest_Rate", F.col("Interest_Rate").cast(FloatType()))
        .withColumn("Num_of_Loan", F.col("Num_of_Loan").cast(FloatType()))
        .withColumn("Type_of_Loan", F.col("Type_of_Loan").cast(StringType()))
        .withColumn("Delay_from_due_date", F.col("Delay_from_due_date").cast(IntegerType()))
        .withColumn("Num_of_Delayed_Payment", F.col("Num_of_Delayed_Payment").cast(IntegerType()))
        .withColumn("Changed_Credit_Limit", F.col("Changed_Credit_Limit").cast(FloatType()))
        .withColumn("Num_Credit_Inquiries", F.col("Num_Credit_Inquiries").cast(FloatType()))
        .withColumn("Credit_Mix", F.col("Credit_Mix").cast(StringType()))
        .withColumn("Outstanding_Debt", F.col("Outstanding_Debt").cast(FloatType()))
        .withColumn("Credit_Utilization_Ratio", F.col("Credit_Utilization_Ratio").cast(FloatType()))
        .withColumn("Credit_History_Age", F.col("Credit_History_Age").cast(IntegerType()))
        .withColumn("Payment_of_Min_Amount", F.col("Payment_of_Min_Amount").cast(StringType()))
        .withColumn("Total_EMI_per_month", F.col("Total_EMI_per_month").cast(FloatType()))
        .withColumn("Amount_invested_monthly", F.col("Amount_invested_monthly").cast(FloatType()))
        .withColumn("Payment_Behaviour", F.col("Payment_Behaviour").cast(StringType()))
        .withColumn("Monthly_Balance", F.col("Monthly_Balance").cast(FloatType()))
        .withColumn("snapshot_date", F.to_date("snapshot_date", "yyyy-MM-dd"))
    )

    
    # Save silver table
    partition_name = f"silver_financials_{snapshot_date_str.replace('-', '_')}.parquet"
    filepath = os.path.join(silver_financials_directory, partition_name)
    df.write.mode("overwrite").parquet(filepath)
    print(f'Saved silver financials to: {filepath}, row count: {df.count()}')
    return df




def process_silver_clickstream(snapshot_date_str, bronze_features_dir, silver_features_dir, spark):
    bronze_file_path = os.path.join(bronze_features_dir, "clickstream", f"bronze_clickstream_{snapshot_date_str.replace('-', '_')}.csv")
    silver_clickstream_directory = os.path.join(silver_features_dir, "clickstream")
    if not os.path.exists(silver_clickstream_directory):
        os.makedirs(silver_clickstream_directory)

    df = spark.read.csv(bronze_file_path, header=True, inferSchema=True)
    print(f'Loaded clickstream from: {bronze_file_path}, row count: {df.count()}')

    # -------------------------------
    # Data Cleaning
    # -------------------------------

    df = ( 
        df
        # 0. Customer_ID: ensure starts with 'CUS_'
        .withColumn("Customer_ID", F.when(F.col("Customer_ID").startswith("CUS_"), F.col("Customer_ID"))
                                      .otherwise(F.concat(F.lit("CUS_"), F.col("Customer_ID"))))
    )

    # -------------------------------
    # Data Type Casting
    # -------------------------------

    # Cast all fe_X columns to IntegerType
    for i in range(1, 21):
        col_name = f"fe_{i}"
        if col_name in df.columns:
            df = df.withColumn(col_name, F.col(col_name).cast(IntegerType()))
    
    
    df = df.withColumn("Customer_ID", F.col("Customer_ID").cast(StringType()))
    df = df.withColumn("snapshot_date", F.to_date(F.col("snapshot_date"), "yyyy-MM-dd"))

    
    # Save silver table
    partition_name = f"silver_clickstream_{snapshot_date_str.replace('-', '_')}.parquet"
    filepath = os.path.join(silver_clickstream_directory, partition_name)
    df.write.mode("overwrite").parquet(filepath)
    print(f'Saved silver clickstream to: {filepath}, row count: {df.count()}')
    return df