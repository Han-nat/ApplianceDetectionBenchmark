import math


def perc_change_last_transmitted(df, column='aggregate', threshold=0.1):
    if column not in df.columns:
        raise ValueError(f'Column {column} not found in dataframe')

    latest_val = df[column].iloc[0]
    ids = [0]

    for i in range(1, len(df)):
        recent_val = df[column].iloc[i]
        if latest_val != 0:
            perc = 100 * (recent_val - latest_val) / latest_val
        else:
            perc = math.inf

        if abs(perc) > threshold:
            ids.append(i)

    return df.iloc[ids]
