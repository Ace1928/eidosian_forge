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
def normalize_whitespaces(sentence):
    return MULTI_WHITESPACES_REGEX.sub(' ', sentence).strip()