import parlai.core.build_data as build_data
import os
import subprocess
import shutil
import csv
import time
from parlai.core.build_data import DownloadableFile
def read_csv_to_dict_list(filepath):
    f = open(filepath, 'r')
    return (csv.DictReader(f, delimiter=','), f)