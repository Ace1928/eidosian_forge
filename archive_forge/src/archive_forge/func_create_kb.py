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
def create_kb(vocab):
    kb = InMemoryLookupKB(vocab, entity_vector_length=1)
    kb.add_entity('test', 0.0, zeros((1,), dtype='f'))
    return kb