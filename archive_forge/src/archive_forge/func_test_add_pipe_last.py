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
@pytest.mark.parametrize('name1,name2', [('parser', 'lambda_pipe')])
def test_add_pipe_last(nlp, name1, name2):
    Language.component('new_pipe2', func=lambda doc: doc)
    nlp.add_pipe('new_pipe2', name=name2)
    nlp.add_pipe('new_pipe', name=name1, last=True)
    assert nlp.pipeline[0][0] != name1
    assert nlp.pipeline[-1][0] == name1