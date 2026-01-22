import torch
import numpy as np
import scipy.sparse as sp
import argparse
import os
import math
from multiprocessing import Pool as ProcessPool
from multiprocessing.util import Finalize
from functools import partial
from collections import Counter
from . import utils
from .doc_db import DocDB
from . import tokenizers
import parlai.utils.logging as logging
def live_count_matrix_t(args, cands):
    global PROCESS_TOK
    if PROCESS_TOK is None:
        PROCESS_TOK = tokenizers.get_class(args.tokenizer)()
    row, col, data = ([], [], [])
    for i, c in enumerate(cands):
        cur_row, cur_col, cur_data = count_text(args.ngram, args.hash_size, i, c)
        row += cur_row
        col += cur_col
        data += cur_data
    count_matrix = torch.sparse.FloatTensor(torch.LongTensor([row, col]), torch.FloatTensor(data), torch.Size([args.hash_size, len(cands)])).coalesce()
    return count_matrix