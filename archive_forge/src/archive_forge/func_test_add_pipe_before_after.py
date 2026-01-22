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
def test_add_pipe_before_after():
    """Test that before/after works with strings and ints."""
    nlp = Language()
    nlp.add_pipe('ner')
    with pytest.raises(ValueError):
        nlp.add_pipe('textcat', before='parser')
    nlp.add_pipe('textcat', before='ner')
    assert nlp.pipe_names == ['textcat', 'ner']
    with pytest.raises(ValueError):
        nlp.add_pipe('parser', before=3)
    with pytest.raises(ValueError):
        nlp.add_pipe('parser', after=3)
    nlp.add_pipe('parser', after=0)
    assert nlp.pipe_names == ['textcat', 'parser', 'ner']
    nlp.add_pipe('tagger', before=2)
    assert nlp.pipe_names == ['textcat', 'parser', 'tagger', 'ner']
    with pytest.raises(ValueError):
        nlp.add_pipe('entity_ruler', after=1, first=True)
    with pytest.raises(ValueError):
        nlp.add_pipe('entity_ruler', before='ner', after=2)
    with pytest.raises(ValueError):
        nlp.add_pipe('entity_ruler', before=True)
    with pytest.raises(ValueError):
        nlp.add_pipe('entity_ruler', first=False)