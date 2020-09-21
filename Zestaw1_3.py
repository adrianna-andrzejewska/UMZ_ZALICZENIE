import pandas as pd
# loading pandas library
# loading data and changing float settings
df_data = pd.read_csv(
          'train.tsv',
          sep='\t',
          names=[
           'price', 'nr_rooms', 'meters', 'floors', 'location', 'description'])
pd.options.display.float_format = '{:.2f}'.format
# add csv file
df_description = pd.read_csv('description.csv')
# add columns names
df_description.columns = ['floors', 'name_floor']

# merge tabels - the key is column floors
df_flat_name_floor = pd.merge(df_data, df_description, on=['floors'])

# save to file
df_flat_name_floor.to_csv('out2.csv', header=False, index=None)
