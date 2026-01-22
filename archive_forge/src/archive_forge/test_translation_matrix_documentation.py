from collections import namedtuple
import unittest
import logging
import numpy as np
import pytest
from scipy.spatial.distance import cosine
from gensim.models.doc2vec import Doc2Vec
from gensim import utils
from gensim.models import translation_matrix
from gensim.models import KeyedVectors
from gensim.test.utils import datapath, get_tmpfile
Test that translation gives similar results to traditional inference.

        This may not be completely sensible/salient with such tiny data, but
        replaces what seemed to me to be an ever-more-nonsensical test.

        See <https://github.com/RaRe-Technologies/gensim/issues/2977> for discussion
        of whether the class this supposedly tested even survives when the
        TranslationMatrix functionality is better documented.
        