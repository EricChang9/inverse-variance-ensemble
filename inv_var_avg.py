import numpy as np 
import pandas as pd 

def inverse_variance_weighted_mean(values, variances):
    weights = 1 / variances
    return np.sum(values * weights) / np.sum(weights)

data = pd.read_csv("input_data.csv",index_col=0)

#pivot contains the mean predicted value for by state and horizon
pivot = data.pivot_table(index='location', columns='horizon', values='value', aggfunc='mean')

#merge pivot back into data
pivot = pivot.melt(ignore_index=False).reset_index()
pivot.columns = ['location', 'horizon', 'mean_value']

data = data.merge(pivot, on=['location', 'horizon'], how='left')

data['var'] = (data['value'] - data['mean_value'])**2

data['weighted_mean'] = data.groupby(['location', 'horizon']).apply(
    lambda group: inverse_variance_weighted_mean(group['value'], group['var'])
).reset_index(level=[0, 1], drop=True)

print(data)
# Pivot the DataFrame
pivot = data.pivot_table(
    index='location',
    columns='horizon',
    values='weighted_mean',
    aggfunc='first'  # Since the weighted mean is precomputed
)

print(pivot)








































