#! /usr/bin/env python

import sys
import numpy as np
import pandas as pd
from ast import literal_eval

# Open txt file / read lines
fo = open(sys.argv[1], 'r')
lines = fo.readlines()

# Create excel output
writer = pd.ExcelWriter(sys.argv[2])

corr_lines = [i for i in lines if 'Correlation' in i]

configuration_names = [i.split('@')[1] for i in corr_lines]
configuration_names = list(set(configuration_names))

# lists used for data frame structure
rho = []
tau = []
cd = []
var = []
fid = []

# Gather metrics for each configuration
for cn in configuration_names:
    # Create tupples with (metric_name, result) for each unique configuration
    corr_tup = [(l.split('!')[1], literal_eval(l.split(':')[1].strip())) for l in corr_lines if
               l.split('@')[1] == cn]

    corr_dict = dict(corr_tup)
    rho.append(corr_dict['R'])
    tau.append(corr_dict['T'])
    cd.append(corr_dict['CD'])
    var.append(corr_dict['Var'])

corr_res = {'Confs': configuration_names, 'R': rho, 'T': tau, 'CD': cd, 'Var':
        var}
corr_df = pd.DataFrame(corr_res)
corr_df.to_excel(writer, 'Rank Correlation')

writer.save()
