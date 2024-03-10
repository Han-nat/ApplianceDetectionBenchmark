import pandas as pds
import math

DATA_PATH = 'data/UKDALE/'

TEST_INDEX = [2]
APPLIANCE_CASES = ['microwave', 'kettle', 'washing_machine']


def perc_change_last_transmitted(df, column='aggregate', threshold=0.1):
    if column not in df.columns:
        raise ValueError(f'Column {column} not found in dataframe')

    latest_val = df[column].iloc[0]
    ids = [0]

    for i in range(1, len(df)):
        recent_val = df[column].iloc[i]
        if latest_val != 0:
            perc = (recent_val - latest_val) / latest_val
        else:
            perc = math.inf

        if abs(perc) > threshold:
            ids.append(i)
            latest_val = recent_val

    return df.iloc[ids]


def perc_change_prev_values(df, column='aggregate', threshold=0.1):
    if column not in df.columns:
        raise ValueError(f'Column {column} not found in dataframe')

    # Calculate the percentage change
    df['perc_change'] = df[column].pct_change()
    df['perc_change'] = df['perc_change'].fillna(float('inf'))

    mask = (df['perc_change'] > threshold) | (df['perc_change'] < -threshold)

    return df[mask].drop(columns=['perc_change'])


def get_downsampled_dataset(house_indices, appliances, threshold, downsampling_func=perc_change_last_transmitted):
    for house_index in house_indices:
        house_path = DATA_PATH + 'house_' + str(house_index) + '/'

        # House labels
        house_label = pds.read_csv(house_path + 'labels.dat', sep=' ', header=None)
        house_label.columns = ['id', 'appliance_name']

        for appliance in appliances:
            if appliance == 'aggregate':
                continue

            # Aggregated data
            house_data = pds.read_csv(house_path + 'channel_1.dat', sep=' ', header=None)
            house_data.columns = ['time', 'aggregate']
            house_data['time'] = pds.to_datetime(house_data['time'], unit='s')
            house_data = house_data.set_index('time')

            if len(house_label.loc[house_label['appliance_name']==appliance]['id'].values) != 0:
                appliance_index = house_label.loc[house_label['appliance_name'] == appliance]['id'].values[0]

                appl_data = pds.read_csv(house_path + 'channel_' + str(appliance_index) + '.dat', sep=' ', header=None)
                appl_data.columns = ['time', appliance]
                appl_data['time'] = pds.to_datetime(appl_data['time'], unit='s')
                appl_data = appl_data.set_index('time')

                data = pds.merge(house_data, appl_data, on='time', how='inner')

    data = downsampling_func(data, threshold=threshold)

    return data


# you need to get the appliance id from the name found in labels.dat
if __name__ == "__main__":
    for index in TEST_INDEX:
        house_path = DATA_PATH + 'house_' + str(index) + '/'

        # House labels
        house_label = pds.read_csv(house_path + 'labels.dat', sep=' ', header=None)
        house_label.columns = ['id', 'appliance_name']

        # Aggregated data
        house_data = pds.read_csv(house_path + 'channel_1.dat', sep=' ', header=None)
        house_data.columns = ['time', 'aggregate']

        for appliance in APPLIANCE_CASES:
            channel = house_path+'channel_'+str(index)+'.dat'
            appliance_index = house_label.loc[house_label['appliance_name'] == appliance]['id'].values[0]

            appl_data = pds.read_csv(house_path + 'channel_' + str(appliance_index) + '.dat', sep=' ', header=None)
            appl_data.columns = ['time', appliance]

            data = pds.merge(house_data, appl_data, on='time', how='inner')

            data = perc_change_prev_values(data, threshold=0.01)

            print(data.shape)

            #channel_data = data[['time', appliance]]
            #channel_data.to_csv(house_path+'channel_'+str(appliance_index)+'_en_2.dat', sep=' ', header=None, index=None)

            #channel_data = data[['time', 'aggregate']]
            #channel_data.to_csv(house_path+f'channel_1_{appliance}_en_2.dat', sep=' ', header=None, index=None)

