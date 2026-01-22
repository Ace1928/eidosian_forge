import logging
import numpy as np
from numpy import ones, vstack, float32 as REAL
import gensim.models._fasttext_bin
from gensim.models.word2vec import Word2Vec
from gensim.models.keyedvectors import KeyedVectors, prep_vectors
from gensim import utils
from gensim.utils import deprecated
from gensim.models import keyedvectors  # noqa: E402
def save_facebook_model(model, path, encoding='utf-8', lr_update_rate=100, word_ngrams=1):
    """Saves word embeddings to the Facebook's native fasttext `.bin` format.

    Notes
    ------
    Facebook provides both `.vec` and `.bin` files with their modules.
    The former contains human-readable vectors.
    The latter contains machine-readable vectors along with other model parameters.
    **This function saves only the .bin file**.

    Parameters
    ----------
    model : gensim.models.fasttext.FastText
        FastText model to be saved.
    path : str
        Output path and filename (including `.bin` extension)
    encoding : str, optional
        Specifies the file encoding. Defaults to utf-8.

    lr_update_rate : int
        This parameter is used by Facebook fasttext tool, unused by Gensim.
        It defaults to Facebook fasttext default value `100`.
        In very rare circumstances you might wish to fiddle with it.

    word_ngrams : int
        This parameter is used by Facebook fasttext tool, unused by Gensim.
        It defaults to Facebook fasttext default value `1`.
        In very rare circumstances you might wish to fiddle with it.

    Returns
    -------
    None
    """
    fb_fasttext_parameters = {'lr_update_rate': lr_update_rate, 'word_ngrams': word_ngrams}
    gensim.models._fasttext_bin.save(model, path, fb_fasttext_parameters, encoding)