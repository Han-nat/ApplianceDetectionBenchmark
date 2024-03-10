import pandas as pds
from methods.every_n import every_n
from methods.perc_change_prev import perc_change_prev_values

df = pds.read_csv('../data/CER/data/xT_residential_25728.csv', header=1, index_col='id_pdl')

df = perc_change_prev_values(df, 1)

print(df.head())