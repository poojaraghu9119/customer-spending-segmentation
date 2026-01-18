# Importing the required libraries
import pandas as pd
import numpy as np
from pathlib import Path

def feature_eng(df):
    """This function takes the raw dataset as the input, converts the dtype of the column "delivered_at" 
       into datetime, removes the future dates (like above Jan 01, 2026), and creates new features like
       recency (how many days back did each customer made the last purchase), frequency (how many times
       each customer purchases or places an order), monetary (the total aount each customer spends),
       average_order_value(the amount a customer spends for a particular order), avg_items_per_order (the
       average number of items a customer buys per order), customer_lifespan_days (the total number of
       days a customer has been purchasing in the particular company), active months (number of months
       when purchase >= 1), monetary_per_active_month, frequency_per_active_month, inter_purchase_gap_mean,
       inter_purchase_gap_std."""
    
    # Converting the dtype of delivered_at to datetime.
    df["delivered_at"] = pd.to_datetime(df["delivered_at"], format="mixed", utc=True, errors="coerce")

    # Removing the rows which have future dates (if any):
    cutoff_date = pd.Timestamp("2026-01-01", tz="UTC")
    df = df[df["delivered_at"] <= cutoff_date]

    # Creating the monetary feature:
    monetary = (df.groupby("user_id", as_index=False).agg(monetary=("sale_price", "sum")))

    # Creating the frequency feature:
    frequency_df = df.groupby("user_id", as_index = False).agg(frequency = ("order_id", "nunique"))
    frequency_df

    # To create the recency feature, we first create the last purchase date per user_id and order_id, 
    # then we create the last purchase date per customer. Then we have to find the latest purchase date in the entire
    # dataset, which is the last puchase date of the column + 1 day.

    # Creating order-level data
    order_delivery = (df.groupby(["user_id", "order_id"], as_index=False)
                      .agg(order_delivered_at=("delivered_at", "max")))
    
    # Creating customer-level data
    last_purchase = (order_delivery.groupby("user_id", as_index=False)
                     .agg(last_purchase_date=("order_delivered_at", "max")))
    
    reference_date = order_delivery["order_delivered_at"].max() + pd.Timedelta(days=1)

    # Finding the recency of each customer
    last_purchase["recency"] = (reference_date - last_purchase["last_purchase_date"]).dt.days

    # Merging the three dataframes
    rfm_df = monetary.merge(frequency_df, on = "user_id").merge(last_purchase, on = "user_id")

    # Finding the average items per order per customer, that is the average number of items a customer 
    # buys per order. For this we create order-level data first, that is the number of items per order
    items_per_order = df.groupby(["user_id", "order_id"], as_index = False).agg(items_in_order = ("product_id", "count"))
    
    # Now creating the customer-level data to find the avg_items_per_order:
    avg_items_per_order = items_per_order.groupby("user_id", as_index = False).agg(avg_items_per_order = ("items_in_order", "mean"))
    
    # Merging the column with rfm_df:
    rfm_df = pd.merge(rfm_df, avg_items_per_order, on = "user_id")

    # Finding the Average Order Value for each customer, that is the total revenue divided by the 
    # total number of orders. We can do that by dividing the monetary by frequency
    rfm_df["average_order_value"] = rfm_df["monetary"]/rfm_df["frequency"]

    # Find the lifespan of each customer. To find that, we have to first find the first purchase date
    # and the last purchase date per customer and subtract these two dates:
    customer_dates = (order_delivery.groupby("user_id", as_index=False).agg(
        first_purchase_date=("order_delivered_at", "min"),
        last_purchase_date=("order_delivered_at", "max")))
    
    customer_dates["customer_lifespan_days"] = (customer_dates["last_purchase_date"] -
                                                customer_dates["first_purchase_date"]).dt.days + 1
    
    # Now merging customer_lifespan_days with the rfm_df dataset:
    rfm_df = pd.merge(rfm_df, customer_dates, on = "user_id")

    # Now two columns names "last_purchase_date_x" and "last_purchase_date_y" would have been generated.
    # Renaming them
    rfm_df.rename(columns = {"last_purchase_date_x": "last_purchase_date", 
                             "last_purchase_date_y": "last_purchase"}, inplace = True)
    
    # Creating the active_months column
    df["year_month"] = df["delivered_at"].dt.to_period("M") # creating year_month column first which returns date in the format
                                                            # 2024
    # Creating a dataframe active_months_df which contains the active_months column
    active_months_df = (df.groupby("user_id", as_index=False).agg(active_months=("year_month", "nunique")))
    
    # Merging the above dataframe with rfm_df
    rfm_df = pd.merge(rfm_df, active_months_df, on = "user_id")

    # Creating monetary_per_active month and frequency_per_active_month
    rfm_df["monetary_per_active_month"] = rfm_df["monetary"]/rfm_df["active_months"]
    rfm_df["frequency_per_active_month"] = rfm_df["frequency"]/rfm_df["active_months"]

    # NEXT WE HAVE TO CREATE INTER_PURCHASE_GAP_MEAN AND INTER_PURCHASE_GAP_STD.
    # For that, first we have to create the order-level delivered dates, since the raw dataset is item-level:
    order_dates = (df.groupby(["user_id", "order_id"], as_index=False).agg(order_date=("delivered_at", "max")))

    # Sorting the data so as to compute the gaps between consecutive orders correctly
    order_dates = order_dates.sort_values(by=["user_id", "order_date"])

    # Computing the inter-purchase gaps in days
    order_dates["inter_purchase_gap"] = (order_dates.groupby("user_id")["order_date"].diff().dt.days)

    # Aggregating the data to customer-level and creating inter_purchase_gap_mean and inter_purchase_gap_std
    gap_features = (order_dates.groupby("user_id", as_index=False).agg(inter_purchase_gap_mean=("inter_purchase_gap", "mean"),
                                                                       inter_purchase_gap_std=("inter_purchase_gap", "std")))
    
    # For customers who had placed only one order, the inter_purchase_gap_mean and inter_purchase_gap_std might be NaN.
    # We can fill them with 0.

    gap_features[["inter_purchase_gap_mean", "inter_purchase_gap_std"]] = gap_features[
        ["inter_purchase_gap_mean", "inter_purchase_gap_std"]].fillna(0)
    
    # Merging the gap_features dataframe with the rfm_df:
    rfm_df = pd.merge(rfm_df, gap_features, on = "user_id")

    return rfm_df

    
    
