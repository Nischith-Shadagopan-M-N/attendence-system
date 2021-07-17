import os
import numpy as np
from random import shuffle
import matplotlib.pyplot as plt
from PIL import Image
import statistics 

print("Loading training data...")
all_data = []

for personName in os.listdir('vectors'):
    for filename in os.listdir('vectors/' + personName):
        vec = np.load('vectors/' + personName + '/' + filename)
        all_data.append((vec, personName, 'dataset/' + personName + '/' + filename[:-4]))

shuffle(all_data)
training_data = all_data[:int(0.9 * len(all_data))]
test_data = all_data[int(0.9 * len(all_data)):]

print("Loaded training data")

for test_vec, test_truth_label, test_filepath in test_data:
    distances = []

    for train_vec, train_truth_label, _ in training_data:
        dist = np.linalg.norm(test_vec - train_vec)
        distances.append((dist, train_truth_label))

    distances.sort()
    print(distances[:5])
    try:
        test_predicted_label = statistics.mode([l for _,l in distances[:5]]) 
    except statistics.StatisticsError as e:
        test_predicted_label = 'Confused'

    plt.imshow(Image.open(test_filepath))
    plt.suptitle('truth_label: ' + str(test_truth_label) + '\n predicted_label: ' + str(test_predicted_label))
    if test_truth_label != test_predicted_label:
        plt.savefig('./misclassified/' + str(len(os.listdir('misclassified'))) + '.png')
    plt.show()

    


    