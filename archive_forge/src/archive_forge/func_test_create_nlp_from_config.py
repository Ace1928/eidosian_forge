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
def test_create_nlp_from_config():
    config = Config().from_str(nlp_config_string)
    with pytest.raises(ConfigValidationError):
        load_model_from_config(config, auto_fill=False)
    nlp = load_model_from_config(config, auto_fill=True)
    assert nlp.config['training']['batcher']['size'] == 666
    assert len(nlp.config['training']) > 1
    assert nlp.pipe_names == ['tok2vec', 'tagger']
    assert len(nlp.config['components']) == 2
    assert len(nlp.config['nlp']['pipeline']) == 2
    nlp.remove_pipe('tagger')
    assert len(nlp.config['components']) == 1
    assert len(nlp.config['nlp']['pipeline']) == 1
    with pytest.raises(ValueError):
        bad_cfg = {'yolo': {}}
        load_model_from_config(Config(bad_cfg), auto_fill=True)
    with pytest.raises(ValueError):
        bad_cfg = {'pipeline': {'foo': 'bar'}}
        load_model_from_config(Config(bad_cfg), auto_fill=True)