from google.cloud import bigquery
import pandas as pd
from pathlib import Path

def fetch_data():
    """This function extracts the required data for the project from the Google bigquery 
    database and saves it in the data/raw folder."""
    client = bigquery.Client(project="customer-spending-segmentation")
    query = """
            select product_id, user_id, order_id, delivered_at, sale_price 
            from bigquery-public-data.thelook_ecommerce.order_items
            where (status = "Complete" and user_id is not null and sale_price > 0 
            and delivered_at is not null)
            """
    df = client.query(query).to_dataframe()
    return df

def save_raw_data(df, output_path):
    """
    Save raw data to CSV format.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    # Fetch data
    df = fetch_data()

    # Define output path
    raw_data_path = Path("data/raw/raw.csv")

    # Save raw dataset
    save_raw_data(df, raw_data_path)

    # Basic sanity checks
    print("Raw data saved successfully!")
    print(f"Shape: {df.shape}")
    print(df.head())