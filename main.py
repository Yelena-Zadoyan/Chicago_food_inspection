import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv('Food_Inspections.csv')
#MODELLING WITHOUT REINSPECTION category
#dropping the 1 na row with missing inspection type
data = data.drop(data[data['Inspection Type'].isna()].index)
data = data[~data['Inspection Type'].str.contains('Re-Inspection', case=False)]
# print(data.shape)
# print(data.dtypes)

# Results - separating the Fail and Pass
data['Results'].unique()
# dropping the results that do not contain pass or fail
data = data.drop(data[~((data['Results'] == 'Pass') | (data['Results'] == 'Fail'))].index)
# print(data['Results'].unique())

data = pd.concat([data, pd.get_dummies(data['Results'])], axis=1)
data = data.drop(['Results', 'Pass'], axis=1)


import Feature_analysis as FA
modified_data = FA.feature_analysis(data)
print(modified_data.isna().sum())

import train_test_validation as Model
Model.model(modified_data)

