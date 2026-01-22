import gc
import numpy
import pytest
from thinc.api import get_current_ops
import spacy
from spacy.lang.en import English
from spacy.lang.en.syntax_iterators import noun_chunks
from spacy.language import Language
from spacy.pipeline import TrainablePipe
from spacy.tokens import Doc
from spacy.training import Example
from spacy.util import SimpleFrozenList, get_arg_names, make_tempdir
from spacy.vocab import Vocab
@pytest.mark.parametrize('n_pipes', [100])
def test_add_lots_of_pipes(nlp, n_pipes):
    Language.component('n_pipes', func=lambda doc: doc)
    for i in range(n_pipes):
        nlp.add_pipe('n_pipes', name=f'pipe_{i}')
    assert len(nlp.pipe_names) == n_pipes