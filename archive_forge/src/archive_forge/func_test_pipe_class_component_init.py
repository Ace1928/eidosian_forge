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
def test_pipe_class_component_init():
    name1 = 'test_class_component1'
    name2 = 'test_class_component2'

    @Language.factory(name1)
    class Component1:

        def __init__(self, nlp: Language, name: str):
            self.nlp = nlp

        def __call__(self, doc: Doc) -> Doc:
            return doc

    class Component2:

        def __init__(self, nlp: Language, name: str):
            self.nlp = nlp

        def __call__(self, doc: Doc) -> Doc:
            return doc

    @Language.factory(name2)
    def factory(nlp: Language, name=name2):
        return Component2(nlp, name)
    nlp = Language()
    for name, Component in [(name1, Component1), (name2, Component2)]:
        assert name in registry.factories
        with pytest.raises(ValueError):
            nlp.add_pipe(Component(nlp, name))
        nlp.add_pipe(name)
        assert name in nlp.pipe_names
        assert nlp.pipe_factories[name] == name
        assert Language.get_factory_meta(name)
        assert nlp.get_pipe_meta(name)
        pipe = nlp.get_pipe(name)
        assert isinstance(pipe, Component)
        assert isinstance(pipe.nlp, Language)
        pipe = nlp.create_pipe(name)
        assert isinstance(pipe, Component)
        assert isinstance(pipe.nlp, Language)