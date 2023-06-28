import tensorflow as tf
import numpy as np
import heapq

from statistics import variance
from scipy.stats import skew, kurtosis

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBRFClassifier



"""

THE OUTPUTS FILES CONSISTING OF SOFTMAX OUTPUTS WILL BE IN THE FORM OF, 

file_Name_Normal = f"{PSR}PSR {SNR}SNR {UAP BASED} {GROUP UAP APPLIED TO} Normal.txt"
file_Name_Adverse = f"{PSR}PSR {SNR}SNR {UAP BASED} {GROUP UAP APPLIED TO} Adverse.txt"

MAKE SURE THAT YOU ADVISE IF ITS CCM, ICM OR MIX BASED UAP IS IN THAT FORMAT
FOR EXAMPLE FOR CCM BASED UAP APPLIED TO ICM GROUP IT WOULD BE

file_Name_Normal = f"{PSR}PSR {SNR}SNR CCM ICM Normal.txt"
file_Name_Adverse = f"{PSR}PSR {SNR}SNR CCM ICM Adverse.txt"

RUN THE SCRIPT WITH THE CORRECT PSR AND SNR, 10 SNR IS USUALLY GOOD
"""

PSR = 0
SNR = 10

file_Name_Normal = f"{PSR}PSR {SNR}SNR CCM CCM Normal.txt"
file_Name_Adverse = f"{PSR}PSR {SNR}SNR CCM CCM Adverse.txt"

softmax_Output_Normal = []
softmax_Output_Adverse = []

with open(file_Name_Normal) as file:
    test = [float(line) for line in file]
    for i in range(int(len(test) / 11)):
        test2 = test[0 + int(i * 11): 11 + int(i * 11)]
        softmax_Output_Normal.append(test2)

with open(file_Name_Adverse) as file:
    test = [float(line) for line in file]
    for i in range(int(len(test) / 11)):
        test2 = test[0 + int(i * 11): 11 + int(i * 11)]
        softmax_Output_Adverse.append(test2)

""" Finding the ratio between second and first highest elements """
ratio_Normal = []
ratio_Adverse = []
for i in range(len(softmax_Output_Normal)):
    two_Largest = heapq.nlargest(2, softmax_Output_Normal[i])
    ratio_Normal.append(two_Largest[1] / two_Largest[0])
    two_Largest = heapq.nlargest(2, softmax_Output_Adverse[i])
    ratio_Adverse.append(two_Largest[1] / two_Largest[0])

""" Finding the variance, skew and kurtosis """

variance_Normal = []
variance_Adverse = []
skewness_Normal = []
skewness_Adverse = []
kurtosis_Normal = []
kurtosis_Adverse = []

for value in softmax_Output_Normal:
    variance_Normal.append(variance(value))
    skewness_Normal.append(skew(value))
    kurtosis_Normal.append(kurtosis(value))

for value in softmax_Output_Adverse:
    variance_Adverse.append(variance(value))
    skewness_Adverse.append(skew(value))
    kurtosis_Adverse.append(kurtosis(value))

""" Obtain confidence score """
""" Calculates the difference between the largest element and the sum of all the other elements"""

conf_Normal = []
conf_Adverse = []
for i in range(len(softmax_Output_Normal)):
    max_Value_Normal = max(softmax_Output_Normal[i])
    sum_Other_Values_Normal = sum(softmax_Output_Normal[i]) - max_Value_Normal
    conf_Normal.append(max_Value_Normal - sum_Other_Values_Normal)

    max_Value_Adv = max(softmax_Output_Adverse[i])
    sum_Other_Values_Adv = sum(softmax_Output_Adverse[i]) - max_Value_Adv
    conf_Adverse.append(max_Value_Adv - sum_Other_Values_Adv)

""" Putting it all in one array """
Normal_Output = softmax_Output_Normal
Adverse_Output = softmax_Output_Adverse

for i in range(len(Normal_Output)):
    Normal_Output[i].append(ratio_Normal[i])
    Normal_Output[i].append(variance_Normal[i])
    Normal_Output[i].append(kurtosis_Normal[i])
    Normal_Output[i].append(skewness_Normal[i])

    Normal_Output[i].append(conf_Normal[i])

    Adverse_Output[i].append(ratio_Adverse[i])
    Adverse_Output[i].append(variance_Adverse[i])
    Adverse_Output[i].append(kurtosis_Adverse[i])
    Adverse_Output[i].append(skewness_Adverse[i])

    Adverse_Output[i].append(conf_Adverse[i])

""" Combining Normal and Adverse arrays together and assigning labels of either normal or adverse. 0 as Normal 1 as Adverse"""

label_Normal = np.zeros((len(Normal_Output),), dtype = int)
label_Adverse = np.ones((len(Adverse_Output),), dtype = int)
all_Labels = np.ravel([label_Normal, label_Adverse])

all_Output = [Normal_Output, Adverse_Output]
all_Output = np.reshape(all_Output, (2 * len(Normal_Output), 16))

""" Put into training and testing """

train_size = 0.8
test_size = 0.2
res = train_test_split(all_Output, all_Labels, train_size = train_size, test_size = test_size, random_state = 1)
train_data, test_data, train_labels, test_labels = res

""" Classifiers """

" KNearestNeighbours "
""" Change amount of neighbours """
n_neighbours = 250
metric = "manhattan"
weight = "uniform"
kNN = KNeighborsClassifier(n_neighbours, metric = metric, weights = weight, )
kNN.fit(train_data, train_labels)
kNN_Predicted = kNN.predict(test_data)
print("K Nearest Neighbours: ", accuracy_score(kNN_Predicted, test_labels), "k = ", n_neighbours)

" Decision Tree "
decision_Tree = DecisionTreeClassifier()
decision_Tree.fit(train_data, train_labels)
decisionTree_Predicted = decision_Tree.predict(test_data)
print("Decision Tree: ", accuracy_score(decisionTree_Predicted, test_labels))

" Random Forest "
""" Change amount of estimators """
estimator = 10
random_Forest = RandomForestClassifier(n_estimators = estimator)
random_Forest.fit(train_data, train_labels)
randomForest_Predicted = random_Forest.predict(test_data)
print("Random Forest: ", accuracy_score(randomForest_Predicted, test_labels), "n = ", estimator)

# min_sample_leaf = 2
# min_sample_split = 10
# random_Forest=RandomForestClassifier(n_estimators=estimator, max_depth = None, min_samples_leaf = min_sample_leaf, min_samples_split = min_sample_split)

" XGBBoost with Random Forest "

XGB_Random_Forest = XGBRFClassifier(n_estimators = estimator)
XGB_Random_Forest.fit(train_data, train_labels)
XGB_Random_Forest_Predicted = XGB_Random_Forest.predict(test_data)
print("XGB Random Forest: ", accuracy_score(XGB_Random_Forest_Predicted, test_labels), "n = ", estimator)

# colsample_bytree = 0.5
# learning_rate = 0.3
# max_depth = 3
# estimator = 50
# subsample = 1
# XGB_Random_Forest = XGBRFClassifier(n_estimators = estimator, subsample=subsample, colsample_bytree = colsample_bytree, max_depth = max_depth, learning_rate = learning_rate)

# ========================================== Starting NN classifier ========================================== #

label_Normal = np.zeros((len(Normal_Output),), dtype = int)
label_Adverse = np.ones((len(Adverse_Output),), dtype = int)
all_Labels = np.ravel([label_Normal, label_Adverse])

all_Output = [Normal_Output, Adverse_Output]
all_Output = np.reshape(all_Output, (2 * len(Normal_Output), 16))

all_Labels_oneHot = tf.keras.utils.to_categorical(all_Labels)
all_Labels = np.reshape(all_Labels_oneHot, (all_Labels_oneHot.shape[0], 1, 2))

""" Put into training and testing """

train_size = 0.8
test_size = 0.2
res = train_test_split(all_Output, all_Labels, train_size = train_size, test_size = test_size, random_state = 1)
train_data, test_data, train_labels, test_labels = res

model = tf.keras.Sequential([
    tf.keras.layers.Reshape((1, 16), input_shape = (16,)),
    tf.keras.layers.Dense(64, activation = 'relu', input_shape = (1, 16)),
    tf.keras.layers.Dense(32, activation = 'relu'),
    tf.keras.layers.Dense(2, activation = 'softmax')
])  #

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])


model.fit(train_data, train_labels, epochs = 100, batch_size = 32, verbose = 1,
          validation_data = (test_data, test_labels), callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 5, verbose = 0, mode = 'auto',
                                         baseline = None, restore_best_weights = False)])

""" val_accuracy is the accuracy for the tested data"""

