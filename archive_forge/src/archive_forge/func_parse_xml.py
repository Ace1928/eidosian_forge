import parlai.core.build_data as build_data
import glob
import gzip
import multiprocessing
import os
import re
import sys
import time
import tqdm
import xml.etree.ElementTree as ET
from parlai.core.build_data import DownloadableFile
def parse_xml(filepath):
    extension = os.path.splitext(filepath)[1]
    if extension == '.gz':
        with gzip.open(filepath, 'r') as f:
            return ET.parse(f)
    else:
        return ET.parse(filepath)