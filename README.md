# Using Intermarket Analysis to try to predict a stock

MSDS692: Data Science Practicum

Author: Brad Eland

Spring 2024

# Purpose:
I want to use intermarket analysis to try to predict the price of one stock based on data and correlations within other stocks.  The current way people many people try to predict a stock is based on looking at one stock and maybe trying to base the price movement on a previous open/close/high/low.  They may only look at one chart as shown below:
 ![image](https://github.com/bradeland/Regis-MSDS-Practicum-I/assets/23301104/cab6ecba-ba3e-4e0e-a25b-ea2a86ea9590)

Intermarket analysis and machine learning can take more data and process it more quickly.  Machine learning can also help to take some of the human emotion out of the equation.  You can take the above chart and turn it into this:
 ![image](https://github.com/bradeland/Regis-MSDS-Practicum-I/assets/23301104/7317b6be-fc10-4e4b-9cf4-1d27bed0d4fb)

Running the program in pycharm:
I used the Pycharm community 2023.1 edition which was robust enough for this modeling. 
I am including a requirements.txt file.  You will need to run this in your python console.
You will need to make sure the applicable libraries have been installed either via import (if you already have them installed).  Some are already native to pycharm/python while others may need to be added within the pycharm interpreter.  
pip install -r requirements.txt
If you’re unable to run anything or get an error, you can manually download anything under my library section within the script by going to the following location within Pycharm (screenshot shown below):

File-->Settings-->Python Interpreter-->Click the + and search for the library-->Install Package
![image](https://github.com/bradeland/Regis-MSDS-Practicum-I/assets/23301104/c6242b28-a073-4efa-ae32-fdc25dc02085)

# Dataset
The dataset was all gathered by using the yfinance library.  The stocks/futures I chose were:

•	LUV (Southwest Airlines)

•	DAL (Delta Airlines)

•	Crude Oil (CL=F)

The values for each of the sets of data contained the following:

•	Date

•	High

•	Low

•	Open

•	Close

•	Volume

# Data Cleaning
I had to merge three datasets together and then choose which value from the above to base the analysis on.  I noticed that there were some dates where NAN values were pulling through.  For instance, on President's Day 2024, there were no values for the airlines stock as shown below:
![image](https://github.com/bradeland/Regis-MSDS-Practicum-I/assets/23301104/f277d6b2-eca8-49e5-857c-6464580d6690)

This is because each countries stocks arent' traded on federal holidays, but I was using a future (crude oil) in my analysis.  These along with other financial assets, like forex (currencies) are traded throughout the year regardless of any federal holidays.   
I decided to remove these values from the prediction dataset by dropping them.

# Predicting Value
When researching it appears that adjusted close is not the best, but normal/nominal closing is the best value to try to get the next day's stock price.
Adjusted closing takes into account any stock dividends and can skew the numbers as shown in the below screenshot.
![image](https://github.com/bradeland/Regis-MSDS-Practicum-I/assets/23301104/180fa170-248c-4bc1-a117-2b4ade4da104)
Conversley the normal close as shown below is the better number to use when doing shorter term stock predictions:
![image](https://github.com/bradeland/Regis-MSDS-Practicum-I/assets/23301104/10cde0e1-7b2b-4daa-8793-49ba8adb4420)

# Model Used
I decided to use sequential modeling from the keras library in trying to predict the stock price for Southwest Airlines.





