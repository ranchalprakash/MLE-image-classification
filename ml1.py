import numpy as np
import csv

with open("P1_data_train.csv", 'r') as numbers:
    n = list(csv.reader(numbers))

with open("P1_labels_train.csv", 'r') as labels:
    l = list(csv.reader(labels))

label = []
for i in l:
    label.append(int(i[0]))

count_five = 0
count_six = 0
for i in label:
    if i == 5:
        count_five += 1
    else:
        count_six += 1

priori_five = count_five/(count_five+count_six)
priori_six = count_six/(count_five+count_six)

x = np.zeros((777, 64))
r = 0
for i in n:
    for c in range(64):
        x[r][c] = int(i[c])
    r+=1

mu5 = np.zeros((1, 64))
mu6 = np.zeros((1, 64))
pointer = 0
for i in x:
    if label[pointer] == 5:
        mu5 += i
    else:
        mu6 += i
    pointer += 1


mu5 = mu5/count_five
mu6 = mu6/count_six

cov5 = np.zeros((64, 64), dtype=np.float64)
cov6 = np.zeros((64, 64), dtype=np.float64)

pointer = 0
for item in x:
    diff = np.zeros((1, 64))
    if label[pointer] == 5:
        diff = item - mu5
        cov5 += np.dot(diff.T, diff)
    else:
        diff = item - mu6
        cov6 += np.dot(diff.T, diff)
    pointer +=1
    

cov5 = cov5/count_five
cov6 = cov6/count_six

cov5_inv = np.linalg.inv(cov5)
cov6_inv = np.linalg.inv(cov6)

cov5_det = np.linalg.det(cov5)
cov6_det = np.linalg.det(cov6)

def discriminant(vec): 
    g5 =  - 0.5*np.log(cov5_det) + np.log(priori_five) - 0.5*np.dot(np.dot((vec - mu5), cov5_inv), (vec-mu5).T)
    g6 =  - 0.5*np.log(cov6_det) + np.log(priori_six) - 0.5*np.dot(np.dot((vec - mu6), cov6_inv), (vec-mu6).T)
    if g5>g6:
        return 5
    else:
        return 6

with open("P1_data_test.csv", 'r') as numbers:
    v = list(csv.reader(numbers))

test_vec = np.zeros((333, 64))
r = 0
for i in v:
    for c in range(64):
        test_vec[r][c] = int(i[c])
    r+=1

with open("P1_labels_test.csv", 'r') as numbers:
    l = list(csv.reader(numbers))


test_label = []
for i in l:
    test_label.append(int(i[0]))


result_label = []
for t in test_vec:
    result_label.append(discriminant(t))

correct = 0
false_six = 0
false_five = 0
true_five = 0
true_six = 0

for i in range(333):
    if test_label[i] - result_label[i] == 0:
        if test_label[i] == 5:
            true_five += 1
        else:
            true_six += 1
        correct += 1
    elif test_label[i] - result_label[i] == -1:
        false_six += 1
    else:
        false_five += 1


confusion_matrix = np.array([[true_five, false_six],[false_five, true_six]])

#finally prints confusion matrix

print(confusion_matrix)