import pandas as pd
# loading pandas library

# loading data from tsv file and add heads to columns
df_data = pd.read_csv(
        'train.tsv',
        sep='\t',
        names=['price', 'nr_rooms', 'meters',
               'floors', 'location', 'description'])
# choosing the price column and calculating the average
# then rounding off 3 decimal places
mean_price = round(df_data['price'].mean(), 3)

# open file and save price as text
with open('out0.csv', 'w') as file:
    file.write(str(mean_price))

# calculate price for square meters and round again
df_data['price_m'] = (df_data['price']/df_data['meters']).round(3)

# choosing rows where rooms count is more than 3 
# and  price for square meter is less than mean price
df_select_data = df_data.loc[
                (df_data['nr_rooms'] >= 3) &
                (df_data['price_m'] <
                 round(float(df_data['price_m'].mean()), 2))][
                 ['nr_rooms', 'price', 'price_m']]
# we don't need index and headers, save to csv
df_select_data.to_csv('out1.csv', header=False, index=None)
