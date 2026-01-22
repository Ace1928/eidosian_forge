from __future__ import division
import gzip
import io
import logging
import unittest
import os
import shutil
import subprocess
import struct
import sys
import numpy as np
import pytest
from gensim import utils
from gensim.models.word2vec import LineSentence
from gensim.models.fasttext import FastText as FT_gensim, FastTextKeyedVectors, _unpack
from gensim.models.keyedvectors import KeyedVectors
from gensim.test.utils import (
from gensim.test.test_word2vec import TestWord2VecModel
import gensim.models._fasttext_bin
from gensim.models.fasttext_inner import compute_ngrams, compute_ngrams_bytes, ft_hash_bytes
import gensim.models.fasttext
def test_save_load_native(self):
    """Test that serialization works end-to-end.  Not crashing is a success."""
    model_name = 'test_ft_saveload_fb.model'
    with temporary_file(model_name):
        load_native().save(model_name)
        model = FT_gensim.load(model_name)
        self.model_structural_sanity(model)
        model.train(list_corpus, total_examples=len(list_corpus), epochs=model.epochs)
        model.save(model_name)
        self.model_structural_sanity(model)