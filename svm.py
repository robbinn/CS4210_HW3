# -------------------------------------------------------------------------
# AUTHOR: Siwen Wang
# FILENAME: svm.py
# SPECIFICATION: Assignment3, problem 7
# FOR: CS 4200- Assignment #3
# TIME SPENT: 2 hrs
# -----------------------------------------------------------*/

# IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas.
# You have to work here only with standard vectors and arrays

# importing some Python libraries
from sklearn import svm
import csv

dbTraining = []
dbTest = []
X_training = []
Y_training = []
c = [1, 5, 10, 100]
degree = [1, 2, 3]
kernel = ["linear", "poly", "rbf"]
decision_function_shape = ["ovo", "ovr"]
highestAccuracy = 0

# reading the data in a csv file
with open('optdigits.tra', 'r') as trainingFile:
    reader = csv.reader(trainingFile)
    for i, row in enumerate(reader):
        X_training.append(row[:-1])
        Y_training.append(row[-1:])

# reading the data in a csv file
with open('optdigits.tes', 'r') as testingFile:
    reader = csv.reader(testingFile)
    for i, row in enumerate(reader):
        dbTest.append(row)

# used to calculate the accuracy
total_length = len(dbTest)
final_c = 0
final_degree = 0
final_kernel = ""
final_shape = ""

# created 4 nested for loops that will iterate through the values of c, degree, kernel, and decision_function_shape
for i in c:  # iterates over c
    for j in degree:  # iterates over degree
        for k in kernel:  # iterates kernel
            for s in decision_function_shape:  # iterates over decision_function_shape

                # Create an SVM classifier that will test all combinations of c, degree, kernel, and
                # decision_function_shape as hyperparameters. For instance svm.SVC(c=1)
                clf = svm.SVC(C=i, kernel=k, degree=j, decision_function_shape=s)

                # Fit Random Forest to the training data
                format_train_y = []

                for n in Y_training:
                    format_train_y.append(n[0])

                clf.fit(X_training, format_train_y)
                # make the classifier prediction for each test sample and start computing its accuracy
                # --> add your Python code here
                # for :
                correct = 0
                for data in dbTest:
                    class_predicted = clf.predict([data[:-1]])
                    if class_predicted == data[-1]:
                        correct += 1

                current_accuracy = correct / total_length

                # check if the calculated accuracy is higher than the previously one calculated. If so, update update
                # the highest accuracy and print it together with the SVM hyperparameters Example: "Highest SVM
                # accuracy so far: 0.92, Parameters: a=1, degree=2, kernel= poly, decision_function_shape = 'ovo'"
                if current_accuracy > highestAccuracy:
                    highestAccuracy = current_accuracy
                    print(f"Highest SVM accuracy so far: {highestAccuracy}, Parameters: c={i}, degree={j}, "
                          f"kernel= {k}, decision_function_shape = '{s}'")
                    final_c = i
                    final_degree = j
                    final_kernel = k
                    final_shape = s

# print the final, highest accuracy found together with the SVM hyperparameters
# Example: "Highest SVM accuracy: 0.95, Parameters: a=10, degree=3, kernel= poly, decision_function_shape = 'ovr'"
print(f"Highest SVM accuracy: {highestAccuracy}, Parameters: c={final_c}, degree={final_degree}, "
      f"kernel= {final_kernel}, decision_function_shape = '{final_shape}'")
