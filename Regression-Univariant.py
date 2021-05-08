# libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# read data
path = 'week_1-ex_1.txt'
# there is no header in csv file (csv not important as .csv ext but is comma separator)
data = pd.read_csv(path,header=None,names=['Population', 'Profit'])

# show data details
# head as pointer of file stop on line 10 from start
print('data=\n', data.head(10))

