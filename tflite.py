import os
import tensorflow as tf
from PIL import Image
import numpy as np
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


interpreter = tf.lite.Interpreter(model_path='model.tflite')
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

cats = []
dogs = []
totalTimes = []
classTime = []

for filename in os.listdir('Images'):
    totalStart = time.time()
    img = Image.open('Images/' + filename)
    img = img.resize((200, 200))
    np_arr = np.asarray(img)

    input_data = np.array(np_arr, dtype=np.float32)
    input_data = np.expand_dims(input_data, axis=0)

    classStart = time.time()
    interpreter.allocate_tensors()
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    end = time.time()
    if output_data[0] > 0.5:
        dogs.append(filename + ' ' + str(output_data[0]))
    else:
        cats.append(filename + ' ' + str(output_data[0]))
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
print('Percentage:', str(cat_corr_p * 100).split('.')[0] + '%', 'Number of correct images:', cat_corr,
      'Number of wrong '
      'images', cat_wrong)
print('Dog Stats')
print('Percentage:', str(dog_corr_p * 100).split('.')[0] + '%', 'Number of correct images:', dog_corr,
      'Number of wrong '
      'images', dog_wrong)

total_p = (dog_corr + cat_corr) / (len(cats) + len(dogs))
print('Total percentage:', str(total_p).split('.')[1] + '%')

print(totalTimes)