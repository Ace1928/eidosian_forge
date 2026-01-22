from parlai.core.teachers import FixedDialogTeacher, DialogTeacher, ParlAIDialogTeacher
from .build import build
import copy
import json
import os
def get_sentence_tokenizer():
    """
    Loads the nltk sentence tokenizer.
    """
    try:
        import nltk
    except ImportError:
        raise ImportError('Please install nltk (e.g. pip install nltk).')
    st_path = 'tokenizers/punkt/{0}.pickle'.format('english')
    try:
        sent_tok = nltk.data.load(st_path)
    except LookupError:
        nltk.download('punkt')
        sent_tok = nltk.data.load(st_path)
    return sent_tok