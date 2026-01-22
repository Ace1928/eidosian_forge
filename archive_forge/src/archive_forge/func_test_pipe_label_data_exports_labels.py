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
@pytest.mark.parametrize('pipe', ['tagger', 'parser', 'ner', 'textcat', 'morphologizer'])
def test_pipe_label_data_exports_labels(pipe):
    nlp = Language()
    pipe = nlp.add_pipe(pipe)
    assert getattr(pipe, 'label_data', None) is not None
    initialize = getattr(pipe, 'initialize', None)
    assert initialize is not None
    assert 'labels' in get_arg_names(initialize)