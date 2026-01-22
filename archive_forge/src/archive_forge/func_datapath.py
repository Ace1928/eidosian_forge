import contextlib
import tempfile
import os
import shutil
from gensim.corpora import Dictionary
from gensim.utils import simple_preprocess
def datapath(fname):
    """Get full path for file `fname` in test data directory placed in this module directory.
    Usually used to place corpus to test_data directory.

    Parameters
    ----------
    fname : str
        Name of file.

    Returns
    -------
    str
        Full path to `fname` in test_data folder.

    Example
    -------
    Let's get path of test GloVe data file and check if it exits.

    .. sourcecode:: pycon

        >>> from gensim.corpora import MmCorpus
        >>> from gensim.test.utils import datapath
        >>>
        >>> corpus = MmCorpus(datapath("testcorpus.mm"))
        >>> for document in corpus:
        ...     pass


    """
    return os.path.join(module_path, 'test_data', fname)