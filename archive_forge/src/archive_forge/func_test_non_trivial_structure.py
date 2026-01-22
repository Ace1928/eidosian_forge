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
def test_non_trivial_structure(self):
    """Test with non-trivial directory structure, shown below:
        .
        ├── 0.txt
        ├── a_folder
        │   └── 1.txt
        └── b_folder
            ├── 2.txt
            ├── 3.txt
            └── c_folder
                └── 4.txt
        """
    dirpath = tempfile.mkdtemp()
    self.write_docs_to_directory(dirpath, '0.txt')
    a_folder = os.path.join(dirpath, 'a_folder')
    os.mkdir(a_folder)
    self.write_docs_to_directory(a_folder, '1.txt')
    b_folder = os.path.join(dirpath, 'b_folder')
    os.mkdir(b_folder)
    self.write_docs_to_directory(b_folder, '2.txt', '3.txt')
    c_folder = os.path.join(b_folder, 'c_folder')
    os.mkdir(c_folder)
    self.write_docs_to_directory(c_folder, '4.txt')
    corpus = textcorpus.TextDirectoryCorpus(dirpath)
    filenames = list(corpus.iter_filepaths())
    base_names = sorted((name[len(dirpath) + 1:] for name in filenames))
    expected = sorted(['0.txt', 'a_folder/1.txt', 'b_folder/2.txt', 'b_folder/3.txt', 'b_folder/c_folder/4.txt'])
    expected = [os.path.normpath(path) for path in expected]
    self.assertEqual(expected, base_names)
    corpus.max_depth = 1
    self.assertEqual(expected[:-1], base_names[:-1])
    corpus.min_depth = 1
    self.assertEqual(expected[2:-1], base_names[2:-1])
    corpus.max_depth = 0
    self.assertEqual(expected[2:], base_names[2:])
    corpus.pattern = '4.*'
    self.assertEqual(expected[-1], base_names[-1])