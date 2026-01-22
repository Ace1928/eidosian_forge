import sys
from csv import DictReader
from collections import OrderedDict
import json
def new_sub_map(row, sm_dict):
    num_colors = int(row['NumOfColors'])
    sm_dict[num_colors] = OrderedDict()
    sub_map = sm_dict[num_colors]
    sub_map['NumOfColors'] = num_colors
    sub_map['Type'] = row['Type']
    sub_map['Colors'] = [(int(row['R']), int(row['G']), int(row['B']))]
    return sub_map