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
def test_select_pipes_errors(nlp):
    for name in ['c1', 'c2', 'c3']:
        nlp.add_pipe('new_pipe', name=name)
        assert nlp.has_pipe(name)
    with pytest.raises(ValueError):
        nlp.select_pipes()
    with pytest.raises(ValueError):
        nlp.select_pipes(enable=['c1', 'c2'], disable=['c1'])
    with pytest.raises(ValueError):
        nlp.select_pipes(enable=['c1', 'c2'], disable=[])
    with pytest.raises(ValueError):
        nlp.select_pipes(enable=[], disable=['c3'])
    disabled = nlp.select_pipes(disable=['c2'])
    nlp.remove_pipe('c2')
    with pytest.raises(ValueError):
        disabled.restore()