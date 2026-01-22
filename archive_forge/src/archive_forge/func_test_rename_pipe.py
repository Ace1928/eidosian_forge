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
@pytest.mark.parametrize('old_name,new_name', [('old_pipe', 'new_pipe')])
def test_rename_pipe(nlp, old_name, new_name):
    with pytest.raises(ValueError):
        nlp.rename_pipe(old_name, new_name)
    nlp.add_pipe('new_pipe', name=old_name)
    nlp.rename_pipe(old_name, new_name)
    assert nlp.pipeline[0][0] == new_name