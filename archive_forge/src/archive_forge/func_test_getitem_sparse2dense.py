import os
import unittest
import random
import shutil
import numpy as np
from scipy import sparse
from gensim.utils import is_corpus, mock_data
from gensim.corpora.sharded_corpus import ShardedCorpus
def test_getitem_sparse2dense(self):
    sp_tmp_fname = self.tmp_fname + '.sparse'
    corpus = ShardedCorpus(sp_tmp_fname, self.data, shardsize=100, dim=self.dim, sparse_serialization=True, sparse_retrieval=False)
    dense_corpus = ShardedCorpus(self.tmp_fname, self.data, shardsize=100, dim=self.dim, sparse_serialization=False, sparse_retrieval=False)
    item = corpus[3]
    self.assertTrue(isinstance(item, np.ndarray))
    self.assertEqual(item.shape, (1, corpus.dim))
    dslice = corpus[2:6]
    self.assertTrue(isinstance(dslice, np.ndarray))
    self.assertEqual(dslice.shape, (4, corpus.dim))
    ilist = corpus[[2, 3, 4, 5]]
    self.assertTrue(isinstance(ilist, np.ndarray))
    self.assertEqual(ilist.shape, (4, corpus.dim))
    d_dslice = dense_corpus[2:6]
    self.assertEqual(dslice.all(), d_dslice.all())
    self.assertEqual(ilist.all(), dslice.all())