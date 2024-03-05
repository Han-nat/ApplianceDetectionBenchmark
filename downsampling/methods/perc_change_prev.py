def perc_change_prev_values(df, n):
    df = df.T

    for col in df.columns:
        df[str(col) + '_perc_change_prev_'] = 100 * df[col].pct_change(periods=1)

    print(df.head(5))

    return df.T