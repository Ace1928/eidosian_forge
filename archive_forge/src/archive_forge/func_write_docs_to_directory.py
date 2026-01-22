from __future__ import unicode_literals
import codecs
import itertools
import logging
import os
import os.path
import tempfile
import unittest
import numpy as np
from gensim.corpora import (bleicorpus, mmcorpus, lowcorpus, svmlightcorpus,
from gensim.interfaces import TransformedCorpus
from gensim.utils import to_unicode
from gensim.test.utils import datapath, get_tmpfile, common_corpus
def write_docs_to_directory(self, dirpath, *args):
    for doc_num, name in enumerate(args):
        with open(os.path.join(dirpath, name), 'w') as f:
            f.write('document %d content' % doc_num)