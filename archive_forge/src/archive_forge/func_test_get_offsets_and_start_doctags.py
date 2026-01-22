from __future__ import with_statement, division
import logging
import unittest
import os
from collections import namedtuple
import numpy as np
from testfixtures import log_capture
from gensim import utils
from gensim.models import doc2vec, keyedvectors
from gensim.test.utils import datapath, get_tmpfile, temporary_file, common_texts as raw_sentences
@unittest.skipIf(os.name == 'nt', 'See another test for Windows below')
def test_get_offsets_and_start_doctags(self):
    lines = ['line1\n', 'line2\n', 'line3\n', 'line4\n', 'line5\n']
    tmpf = get_tmpfile('gensim_doc2vec.tst')
    with utils.open(tmpf, 'wb', encoding='utf8') as fout:
        for line in lines:
            fout.write(utils.any2unicode(line))
    offsets, start_doctags = doc2vec.Doc2Vec._get_offsets_and_start_doctags_for_corpusfile(tmpf, 1)
    self.assertEqual(offsets, [0])
    self.assertEqual(start_doctags, [0])
    offsets, start_doctags = doc2vec.Doc2Vec._get_offsets_and_start_doctags_for_corpusfile(tmpf, 2)
    self.assertEqual(offsets, [0, 12])
    self.assertEqual(start_doctags, [0, 2])
    offsets, start_doctags = doc2vec.Doc2Vec._get_offsets_and_start_doctags_for_corpusfile(tmpf, 3)
    self.assertEqual(offsets, [0, 6, 18])
    self.assertEqual(start_doctags, [0, 1, 3])
    offsets, start_doctags = doc2vec.Doc2Vec._get_offsets_and_start_doctags_for_corpusfile(tmpf, 4)
    self.assertEqual(offsets, [0, 6, 12, 18])
    self.assertEqual(start_doctags, [0, 1, 2, 3])
    offsets, start_doctags = doc2vec.Doc2Vec._get_offsets_and_start_doctags_for_corpusfile(tmpf, 5)
    self.assertEqual(offsets, [0, 6, 12, 18, 24])
    self.assertEqual(start_doctags, [0, 1, 2, 3, 4])
    offsets, start_doctags = doc2vec.Doc2Vec._get_offsets_and_start_doctags_for_corpusfile(tmpf, 6)
    self.assertEqual(offsets, [0, 0, 6, 12, 18, 24])
    self.assertEqual(start_doctags, [0, 0, 1, 2, 3, 4])