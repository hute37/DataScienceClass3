# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 12:31:05 2016

@author: Franziska S.
"""

### import packages
import pandas
import numpy
import seaborn 
import matplotlib.pyplot as plt
import scipy
import statsmodels.api
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
#df_combined.to_csv(path_or_buf="tmp.csv", sep=' ', na_rep='--', float_format=None, columns=None, header=True, index=True, index_label=None, mode='w', encoding=None, compression=None, quoting=None, quotechar='"', line_terminator='\n', chunksize=None, tupleize_cols=False, date_format=None, doublequote=True, escapechar=None, decimal='.')

#set_numeric(return_list_of_numeric_columns(df_combined))
#print (df_combined.describe())

################# DATA MANAGEMENT #######################################################
#########################################################################################

################# LINEAR REGRESSION #####################################################
#########################################################################################

### scatterplot between two variables of interest
def make_scatter_plot(explain, response):
    seaborn.regplot(x=explain, y=response, fit_reg=True, data=df_combined)
    plt.xlabel(str(explain))
    plt.ylabel(str(response))
    plt.title('Association between '+str(explain)+ ' and '+str(response))
    plt.show()

    print (scipy.stats.pearsonr(df_combined[explain], df_combined[response]))

make_scatter_plot("interface_sc", "ddg_per_1000sasa")

### 0-center interface_sc
mean_interface_sc=df_combined["ddg_per_1000sasa"].mean()
df_combined["interface_sc_centered"]=df_combined["interface_sc"]-mean_interface_sc
print(df_combined["interface_sc_centered"].describe())

### create linear regression model
print ("OLS regression model for the association between interface shape complementarity and ddg per 1000 SolventAccessaibleSurfaceArea")
reg1 = smf.ols('interface_sc_centered ~ ddg_per_1000sasa', data=df_combined).fit()
print (reg1.summary())

################# LINEAR REGRESSION #####################################################
#########################################################################################
