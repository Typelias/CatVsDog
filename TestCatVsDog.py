import os

import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

model2 = tf.keras.models.load_model('cat_vs_dog.h5')

model2.summary()

path = 'Images/cat10.jpg'
cats = []
dogs = []

for filename in os.listdir('Images'):
    img = image.load_img('Images/' + filename, target_size=(200, 200))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    classes = model2.predict(images, batch_size=10)
    if classes[0] > 0.5:
        dogs.append(filename + ' ' + str(classes[0]))
    else:
        cats.append(filename+' ' + str(classes[0]))

print('Cats: ' + str(cats))
print('Dogs: ' + str(dogs))