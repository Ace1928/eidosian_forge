import os
import zstandard
import ujson as json
import time
import tarfile
import codecs
from functools import reduce
import jsonlines
import io
from zipfile import ZipFile
import gzip
from math import ceil
import mmap
import multiprocessing as mp
from pathlib import Path
def read_owt_subset(self, file):
    utf8reader = codecs.getreader('utf-8')
    tar = tarfile.open(file, encoding='utf-8')
    for name in tar.getmembers():
        fp = utf8reader(tar.extractfile(name))
        contents = fp.read()
        yield contents