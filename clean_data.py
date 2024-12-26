import pandas as pd
import numpy as np 

#data cleaning and organizational steps 
data = pd.read_csv("ensemble.csv",sep=",",dtype='string')

# Assuming your DataFrame is named 'df'

# Optional: If there are unwanted columns like 'Unnamed: 0', you can drop them
if 'Unnamed: 0' in data.columns:
    data = data.drop(columns=['Unnamed: 0'])

#these are the models that we will be doing the inverse variance weighting 
input_data = data[
    (data['output_type'] == 'point') &
    (data['output_type_id'] == 'mean') &
    (data['method_type'] == 'dev') &
    (~data['method'].str.contains('ensemble', case=False))
]
#these ensemble measurements that we will use to test how well our estimate performs 
benchmark = data[
    (data['output_type'] == 'point') &
    (data['output_type_id'] == 'mean') &
    (data['method_type'] == 'dev') &
    (data['method'].str.contains('ensemble', case=False))
]
#write data to csv to make for faster run time in the future
input_data.to_csv("input_data.csv",index=False)
benchmark.to_csv("benchmark",index=False)





