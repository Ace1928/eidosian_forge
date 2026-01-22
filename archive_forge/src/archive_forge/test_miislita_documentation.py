from __future__ import division  # always use floats
from __future__ import with_statement
import logging
import os
import unittest
from gensim import utils, corpora, models, similarities
from gensim.test.utils import datapath, get_tmpfile

        Make sure we can save and load (un/pickle) TextCorpus objects (as long
        as the underlying input isn't a file-like object; we cannot pickle those).
        