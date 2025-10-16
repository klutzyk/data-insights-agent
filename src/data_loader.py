# data_loader.py - load and preprocess dataset
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Union


def load_df(file_path):
    return pd.read_csv(file_path)

def summarize_df(df: pd.DataFrame) -> str:
    """
    Generate a comprehensive summary of the dataset.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        str: Summary text
    """
    summary_parts = []
    
    # Basic info
    summary_parts.append(f"DATASET SUMMARY")
    summary_parts.append(f"{'='*50}")
    summary_parts.append(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    summary_parts.append(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    summary_parts.append("")
    
    # Missing values
    missing_data = df.isnull().sum()
    missing_pct = (missing_data / len(df)) * 100
    summary_parts.append("MISSING VALUES:")
    for col in df.columns:
        if missing_data[col] > 0:
            summary_parts.append(f"   {col}: {missing_data[col]} ({missing_pct[col]:.1f}%)")
    if missing_data.sum() == 0:
        summary_parts.append("   No missing values found")
    summary_parts.append("")
    
    # Data types
    summary_parts.append("DATA TYPES:")
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        summary_parts.append(f"   {dtype}: {count} columns")
    summary_parts.append("")
    
    # Numeric columns summary
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        summary_parts.append("NUMERIC COLUMNS:")
        for col in numeric_cols:
            summary_parts.append(f"   {col}:")
            summary_parts.append(f"      Mean: {df[col].mean():.2f}")
            summary_parts.append(f"      Std: {df[col].std():.2f}")
            summary_parts.append(f"      Min: {df[col].min():.2f}")
            summary_parts.append(f"      Max: {df[col].max():.2f}")
        summary_parts.append("")
    
    # Categorical columns summary
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        summary_parts.append("CATEGORICAL COLUMNS:")
        for col in categorical_cols:
            unique_count = df[col].nunique()
            top_value = df[col].value_counts().index[0] if len(df[col].value_counts()) > 0 else "N/A"
            summary_parts.append(f"   {col}: {unique_count} unique values, most common: '{top_value}'")
        summary_parts.append("")
    
    return "\n".join(summary_parts)


def summarize_by_group(df: pd.DataFrame, group_cols: List[str], agg_funcs: Dict[str, List[str]]) -> List[str]:
    """
    Generate summaries for grouped data.
    
    Args:
        df (pd.DataFrame): Input dataframe
        group_cols (List[str]): Columns to group by
        agg_funcs (Dict[str, List[str]]): Dictionary mapping column names to list of aggregation functions
        
    Returns:
        List[str]: List of summary strings for each group
    """
    summaries = []
    
    try:
        # Validate group columns exist
        missing_cols = [col for col in group_cols if col not in df.columns]
        if missing_cols:
            summaries.append(f"Columns not found: {missing_cols}")
            return summaries
        
        # Validate aggregation columns exist
        agg_cols = list(agg_funcs.keys())
        missing_agg_cols = [col for col in agg_cols if col not in df.columns]
        if missing_agg_cols:
            summaries.append(f"Aggregation columns not found: {missing_agg_cols}")
            return summaries
        
        # Perform grouping and aggregation
        grouped = df.groupby(group_cols).agg(agg_funcs)
        
        summaries.append(f"GROUPED SUMMARY")
        summaries.append(f"{'='*50}")
        summaries.append(f"Grouped by: {', '.join(group_cols)}")
        summaries.append(f"Number of groups: {len(grouped)}")
        summaries.append("")
        
        # Generate summary for each group
        for i, (group_key, group_data) in enumerate(grouped.iterrows()):
            if len(group_cols) == 1:
                group_name = f"Group: {group_key}"
            else:
                group_name = f"Group: {dict(zip(group_cols, group_key))}"
            
            summaries.append(f"{group_name}")
            
            # Add aggregated values
            for col in agg_cols:
                for func in agg_funcs[col]:
                    value = group_data[(col, func)]
                    if isinstance(value, (int, float)):
                        summaries.append(f"   {col} ({func}): {value:.2f}")
                    else:
                        summaries.append(f"   {col} ({func}): {value}")
            
            summaries.append("")
            
            # Limit to first 10 groups to avoid overwhelming output
            if i >= 9:
                remaining = len(grouped) - 10
                if remaining > 0:
                    summaries.append(f"... and {remaining} more groups")
                break
        
    except Exception as e:
        summaries.append(f"Error in grouped summary: {str(e)}")
    
    return summaries
