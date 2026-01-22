import sys
from csv import DictReader
from collections import OrderedDict
import json
def read_csv_to_dict():
    color_maps = OrderedDict()
    for scheme_type in ('Sequential', 'Diverging', 'Qualitative'):
        color_maps[scheme_type] = OrderedDict()
    with open('colorbrewer_all_schemes.csv', 'r') as csvf:
        csv = DictReader(csvf)
        for row in csv:
            if row['SchemeType']:
                color_maps[row['SchemeType']][row['ColorName']] = OrderedDict()
                current_map = color_maps[row['SchemeType']][row['ColorName']]
                current_submap = new_sub_map(row, current_map)
            elif row['ColorName']:
                current_submap = new_sub_map(row, current_map)
            elif not row['ColorName']:
                current_submap['Colors'].append((int(row['R']), int(row['G']), int(row['B'])))
    return color_maps