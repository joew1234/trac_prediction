# Sale Price Prediction - Heavy Equipment Industry

## Overview
This project was aimed at understanding importance of data cleaning and feature engineering when collecting datasets. As a 6 hour case study, we were tasked with understanding the data and use it to predict the selling price of the heavy machines at auction based on it's usage, equipment type and configuration. We effectively split the tasks and spend first half of the alloted time engineering features based on our domain knowledge. We then developed a baseline Linear regression model which predicted the average selling price of the tractors. Following the CRISP-DM process, we built our own model using the above engineered features and evaluated against the base model. We then iterated over adding new features while evaluating with previous models. The final model predicts the selling price of a tractor based on parameters including Type of enclosure, product size etc.

Due to confidentiality, the data is not made public. The python code is made available in the src file.

## Motivation
Obtaining the dataset, I wanted to understand how effective data cleaning helps regression models perform better. Real world applications have messy data, and it's understood that majority of a Data Scientist's work revolves around Data Cleaning and Feature Engineering of products. Being one of my first data science projects, I wanted to replicate a real world scenario, and how data scientists work in short time frames to get actionable results. With the help of Galvanize, and its instructors, we were able to get the dataset and carryout our ideas.

## Dataset
The data set was obtained by Galvanize. They have around 54 features, combination of categorical and numerical data. We split the data as training and validation by 70%-30%. We were provided with the test data.

## Data Cleaning and Feature Engineering
Data had many missing values and NA's. Created dummy columns signifying missing or NA values, and imputed the values in the original column with abnormally large values or negatives. This allowed us to identify if the missing values have a significant effect, without just imputing it with the mean or some other number. Some columns had incorrect values, which were not considered.

## Model Development
As a team, we decided to use a Linear Regression model. For a baseline model, we tried to predict the selling price of the heavy machinery, based on average selling prices of previous models. Recording the RMS LogE, we iterated through multiple combinations of features to identify best model. Our best model used 8 significant features, out of which 4 features were identified important based on beta values.

## Result and Inference
From a data science perspective, we understood certain limitations of the regression model as well as its ease of interpretability. We figured how using Lasso Regularization can help identify important features and how Lasso Regularization is different from Ridge. We also got a picture of how messy real world data could be, and how feeding in cleaned, normalized data will help you improve the performance of your model.

## Files

* model.py - Compares the performance of models and stores the best model in pickle format

## Rough timeline

* First 3 hours: EDA and Feature Extraction
* Last 3 hours: Model building and Deployment


## Credits
This project would not be possible without the efforts of my fellow teammates Anusha Mohan, Praveen Raman, Jianda Zhou
