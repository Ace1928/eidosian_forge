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
def read_owt(self, file):
    tar = tarfile.open(file, encoding='utf-8')
    utf8reader = codecs.getreader('utf-8')
    for name in tar.getmembers():
        fp = tar.extractfile(name)
        inner_tar = tarfile.open(fileobj=fp, encoding='utf-8')
        for inner_name in inner_tar.getmembers():
            inner_fp = utf8reader(inner_tar.extractfile(inner_name))
            contents = inner_fp.read()
            yield contents