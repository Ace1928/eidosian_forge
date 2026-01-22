import os
import sys
import argparse
import multiprocessing
import lm_dataformat as lmd
import numpy as np
import time
import tqdm
import ftfy
from tokenizer import build_tokenizer
import indexed_dataset
from threading import Semaphore
def yield_from_files(fnames: list, semaphore):
    """
    Iterator over input documents using lm_dataformat. Should be able to handle jsons / texts /
    other compressed formats. Also filters out empty documents.

    :param fnames: list of filenames
    """

    def yielder(fname, semaphore):
        for f in filter(lambda x: x, lmd.Reader(fname).stream_data()):
            semaphore.acquire()
            yield f
    for fname in fnames:
        semaphore.acquire()
        yield from yielder(fname, semaphore)