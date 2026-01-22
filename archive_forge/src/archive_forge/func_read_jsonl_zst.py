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
def read_jsonl_zst(self, file, get_meta=False, autojoin_paragraphs=True, para_joiner='\n\n', key='text'):
    with open(file, 'rb') as fh:
        cctx = zstandard.ZstdDecompressor()
        reader = io.BufferedReader(cctx.stream_reader(fh))
        rdr = jsonlines.Reader(reader)
        yield from handle_jsonl(rdr, get_meta, autojoin_paragraphs, para_joiner, key)