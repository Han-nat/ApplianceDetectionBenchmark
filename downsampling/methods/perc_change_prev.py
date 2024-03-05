def perc_change_prev_values(df, column='aggregate', threshold=0.1):
    if column not in df.columns:
        raise ValueError(f'Column {column} not found in dataframe')

    # Calculate the percentage change
    df['perc_change'] = df[column].pct_change()
    df['perc_change'] = df['perc_change'].fillna(float('inf'))

    mask = (df['perc_change'] > threshold) | (df['perc_change'] < -threshold)

    return df[mask].drop(columns=['perc_change'])