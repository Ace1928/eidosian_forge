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
def read_dat(self, file):
    with open(file, 'rb') as fh:
        cctx = zstandard.ZstdDecompressor()
        reader = cctx.stream_reader(fh)
        while True:
            ln = reader.read(16).decode('UTF-8')
            if not ln:
                break
            ln = int(ln)
            yield reader.read(ln).decode('UTF-8')