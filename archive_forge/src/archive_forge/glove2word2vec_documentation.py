import sys
import logging
import argparse
from gensim import utils
from gensim.utils import deprecated
from gensim.models.keyedvectors import KeyedVectors
Convert `glove_input_file` in GloVe format to word2vec format and write it to `word2vec_output_file`.

    Parameters
    ----------
    glove_input_file : str
        Path to file in GloVe format.
    word2vec_output_file: str
        Path to output file.

    Returns
    -------
    (int, int)
        Number of vectors (lines) of input file and its dimension.

    