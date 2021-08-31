#! /usr/bin/env python

import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize as norm

# Plotting without x-display
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

# Read xls relevance result (pd.DataFrame)
df = pd.read_excel(sys.argv[1], spreadsheet='Rank Correlation')

# Convert to numpy array
var = np.array([i for i in df['Var']])
# Save configuration length
LEN = len(var)
# Scale [0,1]
norm_var = norm(var.reshape(1, LEN), norm='max').reshape(LEN, )

rel_mi = np.array([i for i in df['MI']])
norm_rel_mi = norm(rel_mi.reshape(1, LEN), norm='max').reshape(LEN, )

# Line width
WIDTH = 2.5
import string
az_range = string.ascii_lowercase[1:LEN+1]

plt.plot(range(LEN), norm_var, linestyle='-', linewidth=WIDTH, label='Rank Variation')
plt.scatter(range(LEN), norm_var)
plt.plot(range(LEN), norm_rel_mi, linestyle=':', linewidth=WIDTH, label='Relevance (MI)')
plt.scatter(range(LEN), norm_rel_mi)
plt.xticks(range(LEN), az_range)
plt.legend(loc='best', handlelength=3)
plt.xlabel('Configuration ID', fontsize=13.5)
plt.ylabel('Metric value (scaled)', fontsize=13.5)
plt.savefig(sys.argv[2] + '.pdf', dpi=1000)
