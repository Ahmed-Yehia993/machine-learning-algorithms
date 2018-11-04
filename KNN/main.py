from datetime import datetime
import gzip, pickle
import numpy as np
import matplotlib.pyplot as plt
import operator
import math
import pandas as pd

def open_mage(img1_arr):
    img1_2d = np.reshape(img1_arr, (28, 28))
    plt.subplot(111)
    plt.imshow(img1_2d, cmap=plt.get_cmap('gray'))
    plt.show()

with gzip.open('mnist.pkl.gz','rb') as ff:
    u = pickle._Unpickler( ff )
    u.encoding = 'latin1'
    train, val, test = u.load()


X_train, y_train = train[0], train[1]

def eculidean_destaince(p1,p2):
    if(len(p1) != len(p2)):
        return ;
    summation = 0;
    for i in enumerate(p1):
        x = p1[i[0]]
        y = p2[i[0]]
        summation+= (x-y)**2

    return math.sqrt(summation)

def get_neighbors(training_set_data,training_set_label, test_instance, k):
    distances = []
    length = len(test_instance)-1
    for x in range(len(training_set_data)):
        dist = eculidean_destaince(test_instance, training_set_data[x])
        distances.append(tuple([training_set_label[x],dist]))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

def get_response(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]

def get_accuracy(test_set, predictions):
    correct = 0
    for x in range(len(test_set)):
        if test_set[x] == predictions[x]:
            correct += 1
    return (correct/float(len(test_set))) * 100.0


size_of_training_data = 1000
size_of_test_data = 100

tarning_data = X_train[:size_of_training_data]
tarning_label = y_train[:size_of_training_data]
test_data = X_train[size_of_training_data:size_of_training_data+size_of_test_data]
test_label = y_train[size_of_training_data:size_of_training_data+size_of_test_data]
predictions=[]

for instance in range(len(test_data)):
    start = datetime.now()
    lst = get_neighbors(tarning_data,tarning_label,test_data[instance],5)
    result = get_response(lst)
    predictions.append(result)
    end = datetime.now()
    if instance is 0:
        print("it will take around :" + str(format((end - start)* size_of_test_data )))
    print('complete ' + str(instance) + " actual: "+ str(tarning_label[instance])+">> expected: "+ str(predictions[instance])+ ", it takes: " +str(format(end - start)))

y_actu = pd.Series(tarning_label, name='Actual')
y_pred = pd.Series(predictions, name='Predicted')
df_confusion = pd.crosstab(y_actu, y_pred, margins=True)
print(df_confusion)

accuracy = get_accuracy(test_label, predictions)
print('Accuracy: ' + repr(accuracy) + '%')






