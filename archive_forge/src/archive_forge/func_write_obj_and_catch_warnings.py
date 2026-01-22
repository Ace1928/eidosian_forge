import warnings
from unittest import TestCase
import pytest
import srsly
from numpy import zeros
from spacy.kb.kb_in_memory import InMemoryLookupKB, Writer
from spacy.language import Language
from spacy.pipeline import TrainablePipe
from spacy.vectors import Vectors
from spacy.vocab import Vocab
from ..util import make_tempdir
def write_obj_and_catch_warnings(obj):
    with make_tempdir() as d:
        with warnings.catch_warnings(record=True) as warnings_list:
            warnings.filterwarnings('always', category=ResourceWarning)
            obj.to_disk(d)
            return list(filter(lambda x: isinstance(x, ResourceWarning), warnings_list))