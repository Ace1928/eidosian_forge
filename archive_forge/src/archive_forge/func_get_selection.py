import csv
import copy
from fnmatch import fnmatch
import json
from io import StringIO
def get_selection(jdata, columns):
    if isinstance(jdata, dict):
        jdata = [jdata]
    sub_table = copy.deepcopy(jdata)
    rmcols = set(get_headers(jdata)).difference(columns)
    for entry in sub_table:
        for col in rmcols:
            if col in entry.keys():
                del entry[col]
    return sub_table