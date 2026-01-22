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
def test_pipe_class_component_model_custom():
    name = 'test_class_component_model_custom'
    arch = f'{name}.arch'
    default_config = {'value1': 1, 'model': {'@architectures': arch, 'nO': 0, 'nI': 0}}

    @Language.factory(name, default_config=default_config)
    class Component:

        def __init__(self, nlp: Language, model: Model, name: str, value1: StrictInt=StrictInt(10)):
            self.nlp = nlp
            self.model = model
            self.value1 = value1
            self.name = name

        def __call__(self, doc: Doc) -> Doc:
            return doc

    @registry.architectures(arch)
    def make_custom_arch(nO: StrictInt, nI: StrictInt):
        return Linear(nO, nI)
    nlp = Language()
    config = {'value1': 20, 'model': {'@architectures': arch, 'nO': 1, 'nI': 2}}
    nlp.add_pipe(name, config=config)
    pipe = nlp.get_pipe(name)
    assert isinstance(pipe.nlp, Language)
    assert pipe.value1 == 20
    assert isinstance(pipe.model, Model)
    assert pipe.model.name == 'linear'
    nlp = Language()
    with pytest.raises(ConfigValidationError):
        config = {'value1': '20', 'model': {'@architectures': arch, 'nO': 1, 'nI': 2}}
        nlp.add_pipe(name, config=config)
    with pytest.raises(ConfigValidationError):
        config = {'value1': 20, 'model': {'@architectures': arch, 'nO': 1.0, 'nI': 2.0}}
        nlp.add_pipe(name, config=config)