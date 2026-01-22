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
def train_gensim(bucket=100, min_count=5):
    model = FT_gensim(bucket=bucket, vector_size=5, alpha=0.05, workers=1, sample=0.0001, min_count=min_count)
    model.build_vocab(TOY_SENTENCES)
    model.train(TOY_SENTENCES, total_examples=len(TOY_SENTENCES), epochs=model.epochs)
    return model