import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import sys 
import re
import csv
from scipy import stats
from itertools import groupby
import glob
import collections
from statistics import mean
import numpy as np
from pprint import pprint as p


csv_files = glob.glob("*.csv")

for i, file in enumerate(csv_files):
    print(str(i) + ": " + str(file))

indices = input("Which two indices should be compared").split()
index1, index2 = indices
print(indices)

files = [csv_files[int(index1)], csv_files[int(index2)]]

result_dict = {}

for filename in files:
    result_dict[filename] = {}
    with open(filename, newline='') as csvfile:
        spamreader = csv.reader(csvfile)
        for i, row in enumerate(spamreader):
            if(i == 0): pass
            else:
                result_dict[filename][row[0]] = float(row[2])

p(result_dict)
temp = []
for key in result_dict.keys():
    x = result_dict[key]
    temp.append(list({k: v for k, v in sorted(x.items(), key=lambda item: item[1])}.keys()))

x1, x2 = temp

print(x1)
print(x2)
tau, p_value = stats.kendalltau(x1,x2)



print(tau,p_value)