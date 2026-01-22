import pickle
import pytest
import srsly
from thinc.api import Linear
import spacy
from spacy import Vocab, load, registry
from spacy.lang.en import English
from spacy.language import Language
from spacy.pipeline import (
from spacy.pipeline.dep_parser import DEFAULT_PARSER_MODEL
from spacy.pipeline.senter import DEFAULT_SENTER_MODEL
from spacy.pipeline.tagger import DEFAULT_TAGGER_MODEL
from spacy.pipeline.textcat import DEFAULT_SINGLE_TEXTCAT_MODEL
from spacy.tokens import Span
from spacy.util import ensure_path, load_model
from ..util import make_tempdir
def test_serialize_custom_trainable_pipe():

    class BadCustomPipe1(TrainablePipe):

        def __init__(self, vocab):
            pass

    class BadCustomPipe2(TrainablePipe):

        def __init__(self, vocab):
            self.vocab = vocab
            self.model = None

    class CustomPipe(TrainablePipe):

        def __init__(self, vocab, model):
            self.vocab = vocab
            self.model = model
    pipe = BadCustomPipe1(Vocab())
    with pytest.raises(ValueError):
        pipe.to_bytes()
    with make_tempdir() as d:
        with pytest.raises(ValueError):
            pipe.to_disk(d)
    pipe = BadCustomPipe2(Vocab())
    with pytest.raises(ValueError):
        pipe.to_bytes()
    with make_tempdir() as d:
        with pytest.raises(ValueError):
            pipe.to_disk(d)
    pipe = CustomPipe(Vocab(), Linear())
    pipe_bytes = pipe.to_bytes()
    new_pipe = CustomPipe(Vocab(), Linear()).from_bytes(pipe_bytes)
    assert new_pipe.to_bytes() == pipe_bytes
    with make_tempdir() as d:
        pipe.to_disk(d)
        new_pipe = CustomPipe(Vocab(), Linear()).from_disk(d)
    assert new_pipe.to_bytes() == pipe_bytes