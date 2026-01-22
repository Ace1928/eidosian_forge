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
def get_count_matrix(args, db_opts):
    """
    Form a sparse word to document count matrix (inverted index).

    M[i, j] = # times word i appears in document j.
    """
    global MAX_SZ
    with DocDB(**db_opts) as doc_db:
        doc_ids = doc_db.get_doc_ids()
    tok_class = tokenizers.get_class(args.tokenizer)
    workers = ProcessPool(args.num_workers, initializer=init, initargs=(tok_class, db_opts))
    logging.info('Mapping...')
    row, col, data = ([], [], [])
    step = max(int(len(doc_ids) / 10), 1)
    batches = [doc_ids[i:i + step] for i in range(0, len(doc_ids), step)]
    _count = partial(count, args.ngram, args.hash_size)
    for i, batch in enumerate(batches):
        logging.info('-' * 25 + 'Batch %d/%d' % (i + 1, len(batches)) + '-' * 25)
        for b_row, b_col, b_data in workers.imap_unordered(_count, batch):
            row.extend(b_row)
            col.extend(b_col)
            data.extend(b_data)
            if len(data) > MAX_SZ:
                break
        if len(data) > MAX_SZ:
            logging.info('Reached max indexable size, breaking.')
            break
    workers.close()
    workers.join()
    logging.info('Creating sparse matrix...')
    data, row, col = truncate(data, row, col)
    count_matrix = sp.csr_matrix((data, (row, col)), shape=(args.hash_size, len(doc_ids) + 1))
    count_matrix.sum_duplicates()
    return count_matrix