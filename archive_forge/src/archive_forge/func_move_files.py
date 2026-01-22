import parlai.core.build_data as build_data
import os
import subprocess
import shutil
import csv
import time
from parlai.core.build_data import DownloadableFile
def move_files(base_path, sets=('train', 'valid', 'test')):
    source = os.listdir(base_path)
    for f in source:
        for s in sets:
            if f.endswith('_' + s + '.csv'):
                final_name = f[:-len('_' + s + '.csv')] + '.csv'
                f = os.path.join(base_path, f)
                shutil.move(f, os.path.join(base_path, s, final_name))