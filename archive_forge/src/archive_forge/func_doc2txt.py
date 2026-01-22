from parlai.core.agents import Agent
from parlai.utils.misc import AttrDict
from .doc_db import DocDB
from .tfidf_doc_ranker import TfidfDocRanker
from .build_tfidf import run as build_tfidf
from .build_tfidf import live_count_matrix, get_tfidf_matrix
from numpy.random import choice
from collections import deque
import math
import random
import os
import json
import sqlite3
def doc2txt(self, docid):
    if not self.opt.get('index_by_int_id', True):
        docid = self.ranker.get_doc_id(docid)
    if self.ret_mode == 'keys':
        return self.db.get_doc_text(docid)
    elif self.ret_mode == 'values':
        return self.db.get_doc_value(docid)
    else:
        raise RuntimeError('Retrieve mode {} not yet supported.'.format(self.ret_mode))