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
@pytest.mark.parametrize('name', ['my_component'])
def test_enable_pipes_method(nlp, name):
    nlp.add_pipe('new_pipe', name=name)
    assert nlp.has_pipe(name)
    disabled = nlp.select_pipes(enable=[])
    assert not nlp.has_pipe(name)
    disabled.restore()