#This script uses RAO AllSky images as input and determines which nights are clear through two machine learning algorithms (one for sky condition classification and other to extract the date stamp from the image.)
#Can run images in batch. Simply write the directory the images are in as the argument of line 33.

#***MAKE SURE TO EDIT FILE LOCATIONS FOR YOUR USE.***
directoryOfImages = 'C://Users/zsumn/Desktop/AllSkyTestingSub'
allSkyModelLocation = 'C://Users/zsumn/Desktop/AllSkyML/AllSkyModel'
digitModelLocation = 'C://Users/zsumn/Desktop/AllSkyML/DigitDetection/DigitModel'
saveClearNightInfoLocation = 'C://Users/zsumn/Desktop/ImageCharacteristics.csv'

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import os
from matplotlib.gridspec import GridSpec

#I have already trained the two machine learning models and simply import them for use in this script.
#Please contact me if you would like to train them again.

Allskymodel = tf.keras.models.load_model(allSkyModelLocation)
allsky_classnames = ['Clear', 'Cloudy', 'Day', 'EdgeCloud', 'PartlyCloud', 'WispyCloud']

fig = plt.figure(figsize=(12, 12))
gs = GridSpec(5, 5)

datemodel = tf.keras.models.load_model(digitModelLocation)
date_classnames = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

#Pixel locations of date stamp.

slashboxes = [(5, 5), (11, 5), (17, 5), (23, 5), (32, 5), (38, 5), (47, 5), (53, 5), (5, 18), (11, 18), (20, 18), (26, 18), (35, 18), (41, 18), (5, 465), (11, 465), (17, 465), (23, 465), (29, 465), (35, 465), (41, 465), (47, 465), (53, 465)]
dashboxes = [(5, 5), (11, 5), (17, 5), (23, 5), (33, 5), (39, 5), (49, 5), (55, 5), (5, 18), (11, 18), (20, 18), (26, 18), (35, 18), (41, 18), (5, 465), (11, 465), (17, 465), (23, 465), (29, 465), (35, 465), (41, 465), (47, 465), (53, 465)]

image_characteristics = pd.DataFrame({})


for allsky_filename in os.listdir(directoryOfImages):
    print(allsky_filename)

    #Load in images.
    img = tf.keras.utils.load_img(f'{directoryOfImages}/{allsky_filename}', target_size= (256, 256)) #
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    #Predict sky conditions and associated confidence of prediction
    predictions = Allskymodel.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    correctclass = allsky_classnames[np.argmax(score)]

    #If more than 75% confident sky is clear, read off the time and date from the date stamp.
    if correctclass == 'Clear' and np.max(score)*100 > 75:
        #Load in date stamp.
        dateimg = tf.keras.utils.load_img(f'{directoryOfImages}/{allsky_filename}')
        dateimg_array = tf.keras.utils.img_to_array(dateimg)
        dateimg_array = tf.expand_dims(dateimg_array, 0) # Create a batch

        #Crop image down to date stamp.
        slashcropped = tf.image.crop_to_bounding_box(dateimg_array, 6, 28, 10, 4)
        areaOfInterest = slashcropped[0, :, :, 0]
        
        if areaOfInterest[0][3] > 0.5:
            boxes = slashboxes
        else:
            boxes = dashboxes
        
        j = 0
        numbers = []
        #Read digits of the date individually.
        for pair in boxes:
            #Crop and predict digit.
            cropped = tf.image.crop_to_bounding_box(dateimg_array, pair[1], pair[0], 10, 6)
            datepredictions = datemodel.predict(cropped)
            datescore = tf.nn.softmax(datepredictions[0])

            #If digit classification more than 95% confident, store the number and move to the next. If not, move on to next sky condition classification.
            if np.max(datescore)*100 > 95:
                digit = date_classnames[np.argmax(datescore)]
                numbers.append(digit)
            else:
                print(f'Digit classification uncertain.\n{date_classnames[np.argmax(datescore)]}: {np.max(datescore)*100}')
                break
        #Format and save date to pandas dataframe.
        try:
            date = f'{numbers[0]}{numbers[1]}{numbers[2]}{numbers[3]}/{numbers[4]}{numbers[5]}/{numbers[6]}{numbers[7]}'
            time = f'{numbers[8]}{numbers[9]}:{numbers[10]}{numbers[11]}:{numbers[12]}{numbers[13]}'
            sn = f'{numbers[14]}{numbers[15]}{numbers[16]}{numbers[17]}{numbers[18]}{numbers[19]}{numbers[20]}{numbers[21]}{numbers[22]}'
            print(
                "This image is most likely {} with a {:.2f} percent confidence."
                .format(allsky_classnames[np.argmax(score)], 100 * np.max(score))
            )
            print(f'DATE: {date}.\nTIME: {time}.\nSERIAL NUMBER: {sn}')
            j += 1
            image_characteristics = pd.concat([image_characteristics, pd.DataFrame(data={'SN': sn, 'Date': date, 'Time': time}, index=[j], dtype='str')])

        except IndexError:
            print('Prediction failed.')

#Save times of clear images to pandas dataframe.
image_characteristics.to_csv(saveClearNightInfoLocation)