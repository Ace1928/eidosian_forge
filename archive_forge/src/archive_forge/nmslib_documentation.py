from __future__ import absolute_import
import pickle as _pickle
from smart_open import open
from gensim import utils
from gensim.models.doc2vec import Doc2Vec
from gensim.models.word2vec import Word2Vec
from gensim.models.fasttext import FastText
from gensim.models import KeyedVectors
Find the approximate `num_neighbors` most similar items.

        Parameters
        ----------
        vector : numpy.array
            Vector for a word or document.
        num_neighbors : int
            How many most similar items to look for?

        Returns
        -------
        list of (str, float)
            List of most similar items in the format `[(item, cosine_similarity), ... ]`.

        