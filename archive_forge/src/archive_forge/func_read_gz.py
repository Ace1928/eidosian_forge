from typing import Optional
import gzip
import os
from parlai.core.build_data import DownloadableFile
import parlai.core.build_data as build_data
@classmethod
def read_gz(cls, filename):
    f = gzip.open(filename, 'rb')
    return [x.decode('utf-8') for x in f.readlines()]