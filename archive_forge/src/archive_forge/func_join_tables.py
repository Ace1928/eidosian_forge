import csv
import copy
from fnmatch import fnmatch
import json
from io import StringIO
def join_tables(join_column, jdata, *jtables):
    indexes = []
    for jtable in [jdata] + list(jtables):
        if isinstance(jtable, dict):
            jtable = [jtable]
        index = {}
        [index.setdefault(entry[join_column], entry) for entry in jtable]
        indexes.append(index)
    merged_jdata = []
    for join_id in indexes[0].keys():
        for index in indexes[1:]:
            indexes[0][join_id].update(index[join_id])
            merged_jdata.append(indexes[0][join_id])
    return merged_jdata