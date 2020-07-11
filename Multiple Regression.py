# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 15:09:37 2020

@author: Amogh

                                   #####  PROBLEM/TASK  #####
We are creating a Multiple Regression Model explaining a movie´s revenue (dependent variable)
for movies released between 2010-2016.


"""
# Make sure the .csv file is in the Current Working Directory. Use the following code to confirm the CWD:
# import os
# os.getcwd()
# os.listdir(os.getcwd())

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols


# Preparing the moives data:
movie = pd.read_csv('movies_metadata.csv', low_memory = False)
movie = movie.set_index(pd.to_datetime(movie.release_date, errors = "coerce")).drop(columns = ["release_date"])
movie.sort_index(inplace = True)
df = movie.loc["2010":"2016", ["title", "budget", "revenue", "vote_average", "popularity", 
                               "belongs_to_collection", "original_language"]].copy()
df.budget = pd.to_numeric(df.budget, errors="coerce")
df.popularity = pd.to_numeric(df.popularity, errors= "coerce")
df = df[(df.revenue > 0) & (df.budget > 0)]
df.loc[:, ["budget", "revenue"]] = df.loc[:, ["budget", "revenue"]] / 1000000
df.belongs_to_collection = df.belongs_to_collection.notnull()
df.set_index("title", inplace = True)


              ############   CREATING REGRESSION MODEL   ##########
# Three independent variables; Budget, Popularity and Vote Average are used for the model here.
model = ols("revenue ~ budget + popularity + vote_average", data = df)                
results = model.fit()
print(results.summary())
# COMMENTS: R-squared of 68.4% is good. All three variables are statistically significant. 


# Visualizing relationship between 'belongs_to_collection'(independent variable) and 'revenue'(dependant variable)
plt.figure(figsize = (10, 6))
sns.regplot("belongs_to_collection", "revenue", data = df)
plt.plot()
# Comment: The visualization shows that there is a positive relationship between the two.
#         So if a moive is a part of a franchise/series, we can expect a higher revenue.



        ############   CREATING REGRESSION MODEL (with 'belongs_to_collection' independent variable)   ##########
# Here we add the Dummy Variable 'belongs_to_collection' to the model:
model = ols("revenue ~ budget + popularity + vote_average + belongs_to_collection", data = df)                
results = model.fit()
print(results.summary())
# COMMENTS:
# As we can see, R-Squared and Adj.R-squared has increased to 71.5%
# The coefficient of belongs_to_collection tells us that if a movie belongs to a franchise, we can expect almost 100 Million dollar higher revenue.
# Also, looking at its p-value, we cancan say that the dummay variable 'Belongs_to _collection' is HIGHLY SIGNIFICANT.


# Creating a NEW DUMMY VARIABLE ('original_language_en' - English movie):
df.original_language.nunique()  # to find out no.of unique languages
df_dumm = pd.get_dummies(df, columns = ["original_language"])  # creating columns and dummy vairable for every language


# Creating a regression model between "revenue" and "original_language_en"
model = ols("revenue ~ original_language_en", data = df_dumm)
results = model.fit()
print(results.summary())
# Comments: The variable "original_language_en" seems to be significant when its the only independent variable being used.


 ############   CREATING REGRESSION MODEL (with "original_language_en" independent variable)   ##########
# Here we add the Dummy Variable "original_language_en" to the previous multiple regression model.
model = ols("revenue ~ budget + popularity + vote_average + belongs_to_collection + original_language_en", data = df_dumm)                
results = model.fit()
print(results.summary())
# COMMENTS:
# So here the R-squared doesn't increase.
# Also, the negative slope coefficient and a really High p-value indicates that the variable "original_language_en" is NOT SIGNIFICANT and does not improve the model even if it is significant on a standalone basis.





