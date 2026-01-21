"""
Feature Engineering Module
Reusable functions for creating customer features
"""

import pandas as pd
import numpy as np
from typing import Tuple

def calculate_rfm(orders_df: pd.DataFrame, 
                  order_items_df: pd.DataFrame,
                  analysis_date: pd.Timestamp) -> pd.DataFrame:
    """
    Calculate RFM (Recency, Frequency, Monetary) metrics
    
    Args:
        orders_df: Orders dataframe with delivered orders
        order_items_df: Order items dataframe
        analysis_date: Reference date for recency calculation
        
    Returns:
        DataFrame with RFM metrics per customer
    """
    # Calculate order values
    order_value = order_items_df.groupby('order_id').agg({
        'price': 'sum',
        'freight_value': 'sum'
    }).reset_index()
    order_value['total_value'] = order_value['price'] + order_value['freight_value']
    
    # Merge with orders
    orders_with_value = orders_df.merge(
        order_value[['order_id', 'total_value']], 
        on='order_id', 
        how='left'
    )
    
    # Calculate RFM
    rfm = orders_with_value.groupby('customer_id').agg({
        'order_purchase_timestamp': lambda x: (analysis_date - x.max()).days,
        'order_id': 'count',
        'total_value': 'sum'
    }).reset_index()
    
    rfm.columns = ['customer_id', 'recency', 'frequency', 'monetary']
    
    return rfm


def create_rfm_scores(rfm_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create RFM scores (1-5 scale) and segments
    
    Args:
        rfm_df: DataFrame with recency, frequency, monetary columns
        
    Returns:
        DataFrame with added RFM scores and segments
    """
    df = rfm_df.copy()
    
    # Create scores
    df['r_score'] = pd.qcut(df['recency'], q=5, labels=[5,4,3,2,1], duplicates='drop')
    df['f_score'] = pd.qcut(df['frequency'].rank(method='first'), q=5, labels=[1,2,3,4,5], duplicates='drop')
    df['m_score'] = pd.qcut(df['monetary'], q=5, labels=[1,2,3,4,5], duplicates='drop')
    
    # Segment customers
    def segment_customers(row):
        r = int(row['r_score'])
        f = int(row['f_score'])
        
        if r >= 4 and f >= 4:
            return 'Champions'
        elif r >= 3 and f >= 3:
            return 'Loyal Customers'
        elif r >= 4 and f < 3:
            return 'Potential Loyalists'
        elif r >= 3 and f < 3:
            return 'Promising'
        elif r < 3 and f >= 4:
            return 'At Risk'
        elif r < 3 and f >= 3:
            return 'Need Attention'
        elif r < 2:
            return 'Lost'
        else:
            return 'Others'
    
    df['segment'] = df.apply(segment_customers, axis=1)
    
    return df


def create_temporal_features(orders_df: pd.DataFrame,
                             customer_ids: list) -> pd.DataFrame:
    """
    Create time-based features for customers
    
    Args:
        orders_df: Orders dataframe
        customer_ids: List of customer IDs
        
    Returns:
        DataFrame with temporal features
    """
    timeline = orders_df.groupby('customer_id')['order_purchase_timestamp'].agg([
        ('first_order', 'min'),
        ('last_order', 'max'),
        ('order_count', 'count')
    ]).reset_index()
    
    timeline['customer_lifetime_days'] = (
        timeline['last_order'] - timeline['first_order']
    ).dt.days
    
    timeline['avg_days_between_orders'] = np.where(
        timeline['order_count'] > 1,
        timeline['customer_lifetime_days'] / (timeline['order_count'] - 1),
        0
    )
    
    return timeline


def create_order_value_features(orders_df: pd.DataFrame,
                                order_items_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create order value statistics per customer
    """
    # Calculate order values
    order_value = order_items_df.groupby('order_id')['price'].sum().reset_index()
    order_value.columns = ['order_id', 'total_value']
    
    # Merge
    orders_with_value = orders_df.merge(order_value, on='order_id')
    
    # Aggregate
    value_stats = orders_with_value.groupby('customer_id')['total_value'].agg([
        ('avg_order_value', 'mean'),
        ('min_order_value', 'min'),
        ('max_order_value', 'max'),
        ('std_order_value', 'std')
    ]).reset_index()
    
    value_stats['std_order_value'].fillna(0, inplace=True)
    
    return value_stats


def create_payment_features(payments_df: pd.DataFrame,
                           orders_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create payment behavior features
    """
    payment_data = payments_df.merge(
        orders_df[['order_id', 'customer_id']], 
        on='order_id'
    )
    
    payment_stats = payment_data.groupby('customer_id').agg({
        'payment_type': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'credit_card',
        'payment_installments': 'mean'
    }).reset_index()
    
    payment_stats.columns = ['customer_id', 'preferred_payment_type', 'avg_installments']
    
    return payment_stats


def create_review_features(reviews_df: pd.DataFrame,
                          orders_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create review behavior features
    """
    review_data = reviews_df.merge(
        orders_df[['order_id', 'customer_id']], 
        on='order_id'
    )
    
    review_stats = review_data.groupby('customer_id')['review_score'].agg([
        ('avg_review_score', 'mean'),
        ('min_review_score', 'min'),
        ('max_review_score', 'max')
    ]).reset_index()
    
    return review_stats


def create_churn_target(rfm_df: pd.DataFrame,
                       churn_threshold_days: int = 180) -> pd.DataFrame:
    """
    Create churn target variable
    
    Args:
        rfm_df: DataFrame with recency column
        churn_threshold_days: Days threshold for churn definition
        
    Returns:
        DataFrame with is_churned column
    """
    df = rfm_df.copy()
    df['is_churned'] = (df['recency'] > churn_threshold_days).astype(int)
    return df


def build_feature_matrix(orders_df: pd.DataFrame,
                        order_items_df: pd.DataFrame,
                        payments_df: pd.DataFrame,
                        reviews_df: pd.DataFrame,
                        customers_df: pd.DataFrame,
                        analysis_date: pd.Timestamp,
                        churn_threshold: int = 180) -> pd.DataFrame:
    """
    Build complete feature matrix for modeling
    
    Args:
        orders_df: Orders dataframe (delivered only)
        order_items_df: Order items dataframe
        payments_df: Payments dataframe
        reviews_df: Reviews dataframe
        customers_df: Customers dataframe
        analysis_date: Reference date for analysis
        churn_threshold: Days threshold for churn
        
    Returns:
        Complete feature matrix
    """
    # RFM
    rfm = calculate_rfm(orders_df, order_items_df, analysis_date)
    rfm = create_rfm_scores(rfm)
    
    # Temporal features
    temporal = create_temporal_features(orders_df, rfm['customer_id'].tolist())
    
    # Order value features
    value_features = create_order_value_features(orders_df, order_items_df)
    
    # Payment features
    payment_features = create_payment_features(payments_df, orders_df)
    
    # Review features
    review_features = create_review_features(reviews_df, orders_df)
    
    # Merge all
    features = rfm.copy()
    features = features.merge(temporal, on='customer_id', how='left')
    features = features.merge(value_features, on='customer_id', how='left')
    features = features.merge(payment_features, on='customer_id', how='left')
    features = features.merge(review_features, on='customer_id', how='left')
    
    # Fill missing reviews
    review_cols = ['avg_review_score', 'min_review_score', 'max_review_score']
    features[review_cols] = features[review_cols].fillna(3)
    
    # Create target
    features = create_churn_target(features, churn_threshold)
    
    return features