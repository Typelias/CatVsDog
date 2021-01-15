import os

import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

import time


def correct(arr, sub_str, correct_number):
    corr = 0
    wrong = 0
    for name in arr:
        if sub_str in name:
            corr += 1
        else:
            wrong += 1
    return corr / correct_number, corr, wrong


physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    print('Failed')
    pass

model2 = tf.keras.models.load_model('cat_vs_dog.h5')

model2.summary()

cats = []
dogs = []

totalTimes = []
classTime = []

for filename in os.listdir('Images'):
    totalStart = time.time()
    img = image.load_img('Images/' + filename, target_size=(200, 200))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    classStart = time.time()
    classes = model2.predict(images, batch_size=10)
    end = time.time()
    if classes[0] > 0.5:
        dogs.append(filename + ' ' + str(classes[0]))
    else:
        cats.append(filename + ' ' + str(classes[0]))
    totalTimes.append((end - totalStart))
    classTime.append((end - classStart))

print('Cats: ' + str(cats))
print('Dogs: ' + str(dogs))
totalMean = sum(totalTimes) / len(totalTimes)
classMean = sum(classTime) / len(classTime)
print('Mean for total execution: ' + str(totalMean))
print('Mean for classification execution: ' + str(classMean))
print('Number of dogs', len(dogs), 'Number of cats', len(cats))

cat_corr_p, cat_corr, cat_wrong = correct(cats, 'cat', 10)
dog_corr_p, dog_corr, dog_wrong = correct(dogs, 'dog', 10)
print('Cat Stats')
print('Percentage:', str(cat_corr_p*100).split('.')[0]+'%', 'Number of correct images:', cat_corr, 'Number of wrong '
                                                                                                   'images', cat_wrong)
print('Dog Stats')
print('Percentage:', str(dog_corr_p*100).split('.')[0]+'%', 'Number of correct images:', dog_corr, 'Number of wrong '
                                                                                                   'images', dog_wrong)

total_p = (dog_corr + cat_corr) / (len(cats) + len(dogs))
print('Total percentage:', str(total_p).split('.')[1] + '%')


print(totalTimes)
