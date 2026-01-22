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

    Iterator over input documents using lm_dataformat. Should be able to handle jsons / texts /
    other compressed formats. Also filters out empty documents.

    :param fnames: list of filenames
    