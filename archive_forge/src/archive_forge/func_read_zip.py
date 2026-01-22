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
def read_zip(self, file):
    archive = ZipFile(file, 'r')
    for f in archive.namelist():
        yield archive.read(f).decode('UTF-8')