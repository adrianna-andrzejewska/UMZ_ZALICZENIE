import matplotlib.pyplot as plt
import pandas as pd
# setting to display max 85 columns  because default is 10
pd.set_option('display.max_columns', 85)

# loading data
df_data_schema = pd.read_csv('survey_results_schema.csv', header=0)
# loading next column but this time respondent number is row id too
df_data_public = pd.read_csv(
                 'survey_results_public.csv',
                 header=0,
                 usecols=['Respondent', 'WorkWeekHrs', 'Age'],
                 index_col=['Respondent'])
# delete rows where is Nan
df_data_public.dropna(inplace=True)
df_data_public.dtypes
df_data_public.info()

# check unique values
column_values = df_data_public[['Age']].values.ravel()
unique_values = pd.unique(column_values)
print(unique_values)

# round and check values
df_data_public['Age'].round(0)
column_values = df_data_public['Age'].values.ravel()
unique_values = pd.unique(column_values)
print(unique_values)

df_data_public['WorkWeekHrs'].round(0)
column_values = df_data_public['WorkWeekHrs'].values.ravel().astype('int64')
unique_values = pd.unique(column_values)
print(unique_values)

# choose employes who work 160 hours per week because some people said 
# that they worked more than 160 but this is impossible
df_data_public = df_data_public[df_data_public.WorkWeekHrs < 161]

# change type
df_data_public = df_data_public.astype('int64', copy=False)

plt.plot(df_data_public['Age'],
         df_data_public['WorkWeekHrs'],
         'ro',
         markersize=0.3)
df_data_public_gender = pd.read_csv(
                        'survey_results_public.csv',
                        header=0,
                        usecols=[
                             'Respondent', 'WorkWeekHrs', 'Age', 'Gender'],
                        index_col=['Respondent'])
plt.xlabel('Age')
plt.ylabel('WorkWeekHrs')
plt.show()
# drop nulls
df_data_public_gender.dropna(inplace=True)

# check number of genders
column_values = df_data_public_gender['Gender'].values.ravel()
unique_values = pd.unique(column_values)
print(unique_values)
df_data_public_gender['Gender'] = df_data_public_gender['Gender'].astype(str)
df_data_public_gender.info()

# select only man and woman
df_data_public_gender = df_data_public_gender.loc[
                        (df_data_public_gender['Gender'] == 'Man') |
                        (df_data_public_gender['Gender'] == 'Woman')]
column_values = df_data_public_gender['Gender'].values.ravel()
unique_values = pd.unique(column_values)
print(unique_values)

column_values = df_data_public_gender[['Age']].values.ravel()
unique_values = pd.unique(column_values)
print(unique_values)

df_data_public_gender['WorkWeekHrs'].round(0)
column_values = df_data_public_gender[
                'WorkWeekHrs'].values.ravel().astype('int64')
unique_values = pd.unique(column_values)
print(unique_values)

df_data_public_gender = df_data_public_gender[
                        df_data_public_gender.WorkWeekHrs < 161]
grouped = df_data_public_gender.groupby('Gender')
fig, axes = plt.subplots(grouped.ngroups, sharex=True, figsize=(8, 6))

for i, (Gender, d) in enumerate(grouped):
    ax = d.plot.scatter(x='Age', y='WorkWeekHrs', ax=axes[i], label=Gender)
    ax.set_xlabel('AGE OF THE EMPLOYEE')
    ax.set_ylabel('HOURS COUNT WORKED PER WEEK')
fig.tight_layout()
plt.show()
