from google.cloud import bigquery

print("Script started")

client = bigquery.Client(project="customer-spending-segmentation")
print("BigQuery client created")

query = "SELECT 1 AS test_col"
df = client.query(query).to_dataframe()

print("Query executed")
print(df)
