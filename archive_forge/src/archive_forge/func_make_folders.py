import parlai.core.build_data as build_data
import os
import subprocess
import shutil
import csv
import time
from parlai.core.build_data import DownloadableFile
def make_folders(base_path, sets=('train', 'valid', 'test')):
    for s in sets:
        path = os.path.join(base_path, s)
        if not os.path.exists(path):
            os.mkdir(path)