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
@pytest.mark.issue(11443)
def test_enable_disable_conflict_with_config():
    """Test conflict between enable/disable w.r.t. `nlp.disabled` set in the config."""
    nlp = English()
    nlp.add_pipe('tagger')
    nlp.add_pipe('senter')
    nlp.add_pipe('sentencizer')
    with make_tempdir() as tmp_dir:
        nlp.to_disk(tmp_dir)
        assert spacy.load(tmp_dir, enable=['tagger'], config={'nlp': {'disabled': ['senter']}}).disabled == ['senter', 'sentencizer']
        spacy.load(tmp_dir, enable=['tagger'])
        with pytest.raises(ValueError):
            spacy.load(tmp_dir, enable=['senter'], config={'nlp': {'disabled': ['senter', 'tagger']}})