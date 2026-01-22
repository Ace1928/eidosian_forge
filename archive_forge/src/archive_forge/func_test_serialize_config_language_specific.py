import pytest
from catalogue import RegistryError
from thinc.api import Config, ConfigValidationError
import spacy
from spacy.lang.de import German
from spacy.lang.en import English
from spacy.language import DEFAULT_CONFIG, DEFAULT_CONFIG_PRETRAIN_PATH, Language
from spacy.ml.models import (
from spacy.schemas import ConfigSchema, ConfigSchemaPretrain
from spacy.training import Example
from spacy.util import (
from ..util import make_tempdir
def test_serialize_config_language_specific():
    """Test that config serialization works as expected with language-specific
    factories."""
    name = 'test_serialize_config_language_specific'

    @English.factory(name, default_config={'foo': 20})
    def custom_factory(nlp: Language, name: str, foo: int):
        return lambda doc: doc
    nlp = Language()
    assert not nlp.has_factory(name)
    nlp = English()
    assert nlp.has_factory(name)
    nlp.add_pipe(name, config={'foo': 100}, name='bar')
    pipe_config = nlp.config['components']['bar']
    assert pipe_config['foo'] == 100
    assert pipe_config['factory'] == name
    with make_tempdir() as d:
        nlp.to_disk(d)
        nlp2 = spacy.load(d)
    assert nlp2.has_factory(name)
    assert nlp2.pipe_names == ['bar']
    assert nlp2.get_pipe_meta('bar').factory == name
    pipe_config = nlp2.config['components']['bar']
    assert pipe_config['foo'] == 100
    assert pipe_config['factory'] == name
    config = Config().from_str(nlp2.config.to_str())
    config['nlp']['lang'] = 'de'
    with pytest.raises(ValueError):
        load_model_from_config(config)