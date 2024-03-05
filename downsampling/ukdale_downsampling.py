import pandas as pds
from methods.every_n import every_n

TEST_INDEX = [2]
APPLIANCE_CASES = ['microwave', 'kettle', 'washing_machine']
DATA_PATH = 'data/UKDALE/'

# you need to get the appliance id from the name found in labels.dat

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

        data = every_n(data)

        channel_data = data[['time', appliance]]
        channel_data.to_csv(house_path+'channel_'+str(appliance_index)+'_en_2.dat', sep=' ', header=None, index=None)

        channel_data = data[['time', 'aggregate']]
        channel_data.to_csv(house_path+f'channel_1_{appliance}_en_2.dat', sep=' ', header=None, index=None)

