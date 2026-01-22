import numpy as np
import scipy.sparse as sp
from multiprocessing.pool import ThreadPool
from functools import partial
from . import utils
from . import tokenizers
from parlai.utils.logging import logger
def get_doc_id(self, doc_index):
    """
        Convert doc_index --> doc_id.
        """
    return self.doc_dict[1][doc_index] if self.doc_dict else doc_index