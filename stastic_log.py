from collections import Counter
import sys
import time
import re
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame

def get_log_files(filePath):
    files = os.listdir(filePath)
    log_files = []
    for file in files:
        if(file[:3] != 'tra'):
            continue
        else:
            log_files.append(file)
    return log_files

def convert_digit(a):
    res = 0
    a = a[2:]
    for i in range(len(a)):
        tmp = 0.1/(10**i)
        res += int(a[i])*tmp
    return res

def read_files(file_name):
    file = open(file_name)
    content = file.readlines()   #读出的为列表
    file.close()
    str1 = content[-9]
    str2 = content[-8]
    a=re.findall(r'\d',str1) #在字符串中找到正则表达式所匹配的所有数字，a是一个list
    b=re.findall(r'\d',str2) #在字符串中找到正则表达式所匹配的所有数字，a是一个list
    f1_micro = convert_digit(a)
    f1_macro = convert_digit(b)
    return f1_micro, f1_macro

f1_micro = []
f1_macro = []
filePath = '/p300/flair/resources/taggers/1211/'
log_files = get_log_files(filePath)

for log_file in log_files:
    fmi,fma = read_files(filePath + log_file)
    f1_micro.append(fmi)
    f1_macro.append(fma)

plt.plot(f1_micro)
idx = np.argmax(f1_micro)
idx2 = np.argmin(f1_micro)
print(f1_micro)
print(log_files)
print(f1_micro[idx])
print(f1_micro[idx2])
print(log_files[idx])
print(log_files[idx2])

c = {
    'paras' : log_files,
    'res' : f1_micro}
data=DataFrame(c)
print(data)