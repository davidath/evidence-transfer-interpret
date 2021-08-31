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

# Retrieve relevance and redundancy lines
rel_lines = [i for i in lines if 'Relevance' in i]

red_lines = [i for i in lines if 'Redundancy' in i]

configuration_names = [i.split('@')[1] for i in rel_lines]
configuration_names = list(set(configuration_names))

# lists used for data frame structure
rel_mi = []
rel_f = []
rel_pcorr = []

red_mi = []
red_f = []
red_pcorr = []

# Gather metrics for each configuration
for cn in configuration_names:
    # Create tupples with (metric_name, result) for each unique configuration
    rel_tup = [(l.split('!')[1], literal_eval(l.split(':')[1].strip())) for l in rel_lines if
               l.split('@')[1] == cn]
    red_tup = [(l.split('!')[1], literal_eval(l.split(':')[1].strip())) for l in red_lines if
               l.split('@')[1] == cn]

    rel_dict = dict(rel_tup)
    rel_mi.append(rel_dict['MI'][1])
    rel_f.append(rel_dict['F-test'][1])
    rel_pcorr.append(rel_dict['P-Correlation'][1])

    red_dict = dict(red_tup)
    red_mi.append(red_dict['MI'][1])
    red_f.append(red_dict['F-test'][1])
    red_pcorr.append(red_dict['P-Correlation'][1])

rel_res = {'Confs': configuration_names, 'MI': rel_mi, 'F-test': rel_f,
        'P-Correlation': rel_pcorr}
rel_df = pd.DataFrame(rel_res)
rel_df.to_excel(writer, 'Relevance')

red_res = {'Confs': configuration_names, 'MI': red_mi, 'F-test': red_f,
        'P-Corredation': red_pcorr}
red_df = pd.DataFrame(red_res)
red_df.to_excel(writer, 'Redundancy')

writer.save()
