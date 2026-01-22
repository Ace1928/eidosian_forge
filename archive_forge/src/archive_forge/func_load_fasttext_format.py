import logging
import numpy as np
from numpy import ones, vstack, float32 as REAL
import gensim.models._fasttext_bin
from gensim.models.word2vec import Word2Vec
from gensim.models.keyedvectors import KeyedVectors, prep_vectors
from gensim import utils
from gensim.utils import deprecated
from gensim.models import keyedvectors  # noqa: E402
@classmethod
@utils.deprecated('use load_facebook_vectors (to use pretrained embeddings) or load_facebook_model (to continue training with the loaded full model, more RAM) instead')
def load_fasttext_format(cls, model_file, encoding='utf8'):
    """Deprecated.

        Use :func:`gensim.models.fasttext.load_facebook_model` or
        :func:`gensim.models.fasttext.load_facebook_vectors` instead.

        """
    return load_facebook_model(model_file, encoding=encoding)