# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 18:11:48 2016

@author: Franziska S.
"""

### import packages
import pandas
import numpy
import seaborn 
import matplotlib.pyplot as plt
import scipy
import statsmodels.api as sm
import statsmodels.formula.api as smf

### IMPORT DATA
comp = pandas.read_csv('~/Dropbox/DataScienceSpecialization/DataScienceClass3/comp.csv', low_memory=False, header=0, skiprows=1, sep = ',', skipinitialspace=True, warn_bad_lines=True)
exp = pandas.read_csv('~/Dropbox/DataScienceSpecialization/DataScienceClass3/exp.csv', low_memory=False, header=0, skiprows=0, sep = ',', skipinitialspace=True, warn_bad_lines=True)

################# DATA MANAGEMENT #######################################################
#########################################################################################

### define function that returns row names of numeric columns only
def return_list_of_numeric_columns(data_frame):
    numeric_column_list=[column for column in data_frame.columns if data_frame[column].dtype== 'float64' or data_frame[column].dtype== 'int64' ]
    return numeric_column_list

### SET DATA TYPES ####
### function to convert all columns to numeric except for last description column
def set_numeric(dataframe):
    for term in return_list_of_numeric_columns(dataframe):
        dataframe[str(term)]=pandas.to_numeric(dataframe[str(term)])
        
### convert computational and experimental data
set_numeric(comp)
#print(comp.head(n=10))
set_numeric(exp)

### convert description column to string
comp['description']=comp['description'].astype("str")

### create a common index for both data frames so that they can be merged
comp_description_names=comp['description'].values
#print(comp_description_names)
new_comp_description_list=[pdb[:-5]+".pdb" for pdb in comp_description_names]
#print (len(new_comp_description_list))
comp['description_adj']=new_comp_description_list
comp['description_adj']=comp['description_adj'].astype("str")
#print(comp['description_adj'])

exp_description_names=exp['named'].values
new_exp_description_list=[pdb.split("/")[1] for pdb in exp_description_names]
print (len(new_exp_description_list))
exp['description_adj']=new_exp_description_list
exp['description_adj']=exp['description_adj'].astype("str")
#print(comp['description_adj'])

#### combine data frames
df_combined=pandas.concat([comp, exp], axis=1, join_axes=[exp.index])
#print (df_combined.describe())
df_combined.dropna()
#df_combined.set_index("description_adj")
#df_combined.to_csv(path_or_buf="df_combined.csv", sep=' ', na_rep='--', float_format=None, columns=None, header=True, index=True, index_label=None, mode='w', encoding=None, compression=None, quoting=None, quotechar='"', line_terminator='\n', chunksize=None, tupleize_cols=False, date_format=None, doublequote=True, escapechar=None, decimal='.')

#set_numeric(return_list_of_numeric_columns(df_combined))
#print (df_combined.describe())

### recode Keq response variable into two categories: 0 for Keq <1 and 1 for Keq<=1
### create a new variable Keq_cat
df_combined["Keq_cat"]=df_combined["Keq"]
### collapse numeric variable keq down to two categories in Keq_cat
df_combined["Keq_cat"]=df_combined["Keq_cat"].apply(lambda x: 0 if x < 1 else 1)
### set new variable Keq_cat to categorical
### interestingly, the catergorial variable cannot be set to categorical for the following logistic regression --> follow up on why
#df_combined["Keq_cat"]=df_combined["Keq_cat"].astype('category')
### check that collapse worked
#print (df_combined.columns)


################# DATA MANAGEMENT #######################################################
#########################################################################################


################# LOGISTIC REGRESSION MODEL #####################################################
#########################################################################################

### logistic regression Keq response variable and explanatory ddg_per_1000sasa
lreg1 = smf.logit(formula = 'Keq_cat ~ ddg_per_1000sasa', data = df_combined).fit()
print (lreg1.summary())
# odds ratios
print ("Odds Ratios")
print (numpy.exp(lreg1.params))

# odd ratios with 95% confidence intervals
params = lreg1.params
conf = lreg1.conf_int()
conf['OR'] = params
conf.columns = ['Lower CI', 'Upper CI', 'OR']
print (numpy.exp(conf))

### logistic regression with ddg_per_1000sasa and sc_interface
lreg2 = smf.logit(formula = 'Keq_cat ~ ddg_per_1000sasa + interface_sc', data = df_combined).fit()
print (lreg2.summary())
# odds ratios
print ("Odds Ratios")
print (numpy.exp(lreg2.params))

# odd ratios with 95% confidence intervals
params = lreg2.params
conf2 = lreg2.conf_int()
conf2['OR'] = params
conf2.columns = ['Lower CI', 'Upper CI', 'OR']
print (numpy.exp(conf2))


################# LOGISTIC REGRESSION MODEL #####################################################
#########################################################################################
