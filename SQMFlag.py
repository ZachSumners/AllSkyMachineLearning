#This script determines which days are appropriate to reference SQM data using the sky condition classifier.
#This is important because only long-term clear conditions should be referenced.

#***MAKE SURE TO EDIT FILE LOCATIONS FOR YOUR USE.***

import pandas as pd
from astropy.time import Time

#The file for saved clear night dates. Edit for your own purposes.
observations = pd.read_csv('C:/Users/ronald.sumners/Desktop/ImageCharacteristics.csv')

unique_dates = observations['Date'].unique()
frequencyClear = []
sqmdetection = []

for date in unique_dates:
    #Format date. We need to check both the evening of day x and the morning of day x+1 to see if night is clear for a long period.
    year = date[:4]
    month = date[5:7]
    day = date[8:]
    julianformatting = f'{year}-{month}-{day}T00:00:00'
    t = Time(julianformatting, format='isot')
    jd = t.jd
    daybefore = jd - 1

    datebefore = Time(daybefore, format='jd')
    datebefore.format = 'isot'
    daybefore = f'{datebefore.value[:4]}/{datebefore.value[5:7]}/{datebefore.value[8:10]}'

    #Check how many AllSky images are clear between 6PM of day x and 10am of day x+1.
    times = len(observations.loc[((observations['Date'] == date) & (observations['Time'] < '10:00:00')) | ((observations['Date'] == daybefore) & (observations['Time'] > '18:00:00'))])
    frequencyClear.append(times)

    #If more than 60 images clear (1 hour), say date is clear for a long period. Parameter can be adjusted.
    if times > 60:
        sqmdetection.append(date)

#Dates output are the nights STARTING on that day (6pm of day x to 10am of day x+1).
print('CLEAR NIGHTS FOR SQM REFERENCE:')
print(sqmdetection)
