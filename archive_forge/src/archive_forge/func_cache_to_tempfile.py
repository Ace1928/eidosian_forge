import bisect
import os
import pickle
import re
import tempfile
from functools import reduce
from xml.etree import ElementTree
from nltk.data import (
from nltk.internals import slice_bounds
from nltk.tokenize import wordpunct_tokenize
from nltk.util import AbstractLazySequence, LazyConcatenation, LazySubsequence
@classmethod
def cache_to_tempfile(cls, sequence, delete_on_gc=True):
    """
        Write the given sequence to a temporary file as a pickle
        corpus; and then return a ``PickleCorpusView`` view for that
        temporary corpus file.

        :param delete_on_gc: If true, then the temporary file will be
            deleted whenever this object gets garbage-collected.
        """
    try:
        fd, output_file_name = tempfile.mkstemp('.pcv', 'nltk-')
        output_file = os.fdopen(fd, 'wb')
        cls.write(sequence, output_file)
        output_file.close()
        return PickleCorpusView(output_file_name, delete_on_gc)
    except OSError as e:
        raise ValueError('Error while creating temp file: %s' % e) from e