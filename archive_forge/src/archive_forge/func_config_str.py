from typing import Callable, Iterable, Iterator
import pytest
from thinc.api import Config
from spacy.lang.en import English
from spacy.language import Language
from spacy.training import Example
from spacy.training.loop import train
from spacy.util import load_model_from_config, registry
@pytest.fixture
def config_str():
    return '\n    [nlp]\n    lang = "en"\n    pipeline = ["sentencizer","assert_sents"]\n    disabled = []\n    before_creation = null\n    after_creation = null\n    after_pipeline_creation = null\n    batch_size = 1000\n    tokenizer = {"@tokenizers":"spacy.Tokenizer.v1"}\n\n    [components]\n\n    [components.assert_sents]\n    factory = "assert_sents"\n\n    [components.sentencizer]\n    factory = "sentencizer"\n    punct_chars = null\n\n    [training]\n    dev_corpus = "corpora.dev"\n    train_corpus = "corpora.train"\n    annotating_components = ["sentencizer"]\n    max_steps = 2\n\n    [corpora]\n\n    [corpora.dev]\n    @readers = "unannotated_corpus"\n\n    [corpora.train]\n    @readers = "unannotated_corpus"\n    '