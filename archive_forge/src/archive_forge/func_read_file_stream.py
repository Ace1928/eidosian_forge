import glob
import logging
import os
import shutil
import subprocess
import time
from apache_beam.io import gcsio
def read_file_stream(file_list):
    for path in file_list if not isinstance(file_list, basestring) else [file_list]:
        if path.startswith('gs://'):
            for line in _read_cloud_file_stream(path):
                yield line
        else:
            for line in _read_local_file_stream(path):
                yield line