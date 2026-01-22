import pytest
from thinc.api import ConfigValidationError, Linear, Model
import spacy
from spacy.lang.de import German
from spacy.lang.en import English
from spacy.language import Language
from spacy.pipeline.tok2vec import DEFAULT_TOK2VEC_MODEL
from spacy.tokens import Doc
from spacy.util import SimpleFrozenDict, combine_score_weights, registry
from ..util import make_tempdir
def test_pipe_class_component_model():
    name = 'test_class_component_model'
    default_config = {'model': {'@architectures': 'spacy.TextCatEnsemble.v2', 'tok2vec': DEFAULT_TOK2VEC_MODEL, 'linear_model': {'@architectures': 'spacy.TextCatBOW.v3', 'exclusive_classes': False, 'ngram_size': 1, 'no_output_layer': False}}, 'value1': 10}

    @Language.factory(name, default_config=default_config)
    class Component:

        def __init__(self, nlp: Language, model: Model, name: str, value1: StrictInt):
            self.nlp = nlp
            self.model = model
            self.value1 = value1
            self.name = name

        def __call__(self, doc: Doc) -> Doc:
            return doc
    nlp = Language()
    nlp.add_pipe(name)
    pipe = nlp.get_pipe(name)
    assert isinstance(pipe.nlp, Language)
    assert pipe.value1 == 10
    assert isinstance(pipe.model, Model)