import math

import numpy
import pandas as pds

TRAIN_HOUSE_INDICES = [1, 3, 5]
TEST_HOUSE_INDICES = [2]

THRESHOLD = 0.01

appliance_names = ['kettle', 'microwave', 'dishwasher'] #, 'microwave', 'dishwasher'


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

    #print(df.iloc[ids])
    return df.iloc[ids]


def downsample_house(house_index, output='train', downsample=False):
    path = f'data/ukdale/house_{house_index}/'
    appliances = pds.read_csv(f'{path}/labels.dat',
                              sep=' ',
                              names=['id', 'appliance'],
                              header=None)
    appliance_ids = appliances[appliances['appliance'].isin(appliance_names)]['id'].tolist()

    aggregate = pds.read_csv(f'{path}/channel_1.dat',
                             sep=' ',
                             header=None)

    aggregate.columns = ['time', 'aggregate']
    aggregate['time'] = pds.to_datetime(aggregate['time'], unit='s')
    aggregate = aggregate.set_index('time').resample('6S').mean().reset_index().fillna(method='ffill',
                                                                                        limit=60)
    aggregate['aggregate'] = numpy.around(aggregate['aggregate'], decimals=2)
    for appliance_id in appliance_ids:
        appliance = pds.read_csv(f'{path}/channel_{appliance_id}.dat',
                                 sep=' ',
                                 header=None)
        appliance_name = appliances[appliances['id'] == appliance_id]['appliance'].values[0]
        appliance.columns = ['time', f'appliance_{appliance_name}']
        appliance['time'] = pds.to_datetime(appliance['time'], unit='s')
        appliance = appliance.set_index('time').resample('6S').mean().reset_index().fillna(method='ffill',
                                                                                            limit=60)
        aggregate = pds.merge(aggregate, appliance, on='time', how='inner')

    if downsample:
        aggregate = perc_change_last_transmitted(aggregate, 'aggregate', threshold=THRESHOLD)

    aggregate['key'] = house_index
    aggregate['time'] = aggregate['time'].astype('int64')
    aggregate = aggregate.set_index('time')
    aggregate = aggregate.dropna()
    aggregate.to_csv(f'results/datasets/{i}_{output}.csv')

    return aggregate


for i in TRAIN_HOUSE_INDICES:
    downsampled = downsample_house(i, output='train', downsample=False)


for i in TEST_HOUSE_INDICES:
    downsampled = downsample_house(i, output='test', downsample=True)

