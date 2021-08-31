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
df = pd.read_excel(sys.argv[1], spreadsheet='Relevance')

# Convert to numpy array
acc = np.array([i for i in df['ACC']])
# Save configuration length
LEN = len(acc)
# Scale [0,1]
norm_acc = norm(acc.reshape(1, LEN), norm='max').reshape(LEN, )

nmi = np.array([i for i in df['NMI']])
norm_nmi = norm(nmi.reshape(1, LEN), norm='max').reshape(LEN, )

rel_mi = np.array([i for i in df['MI']])
norm_rel_mi = norm(rel_mi.reshape(1, LEN), norm='max').reshape(LEN, )

rel_f = np.array([i for i in df['F-test']])
norm_rel_f = norm(rel_f.reshape(1, LEN), norm='max').reshape(LEN, )

# Line width
WIDTH = 2.5
import string
az_range = string.ascii_lowercase[:LEN]

plt.plot(range(LEN), norm_acc, linestyle='-', linewidth=WIDTH, label='ACC')
plt.scatter(range(LEN), norm_acc)
plt.plot(range(LEN), norm_nmi, linestyle='--', linewidth=WIDTH, label='NMI')
plt.scatter(range(LEN), norm_nmi)
plt.plot(range(LEN), norm_rel_mi, linestyle=':', linewidth=WIDTH, label='Relevance (MI)')
plt.scatter(range(LEN), norm_rel_mi)
plt.plot(range(LEN), norm_rel_f, linestyle='-.', linewidth=WIDTH, label='Relevance (F-test)')
plt.scatter(range(LEN), norm_rel_f)
plt.xticks(range(LEN), az_range)
plt.legend(loc='best', handlelength=3)
plt.xlabel('Configuration ID', fontsize=13.5)
plt.ylabel('Metric value (scaled)', fontsize=13.5)
plt.savefig(sys.argv[2] + '.pdf', dpi=1000)
