from array import array
from bisect import bisect_right
from os.path import isfile
from struct import Struct
from warnings import warn
from pyzstd.zstdfile import ZstdDecompressReader, ZstdFile, \
def get_frame_sizes(self, i):
    return (self._cumulated_c_size[i - 1], self._cumulated_d_size[i - 1])