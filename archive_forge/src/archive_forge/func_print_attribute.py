import re
import datetime
import numpy as np
import csv
import ctypes
def print_attribute(name, tp, data):
    type = tp.type_name
    if type == 'numeric' or type == 'real' or type == 'integer':
        min, max, mean, std = basic_stats(data)
        print(f'{name},{type},{min:f},{max:f},{mean:f},{std:f}')
    else:
        print(str(tp))