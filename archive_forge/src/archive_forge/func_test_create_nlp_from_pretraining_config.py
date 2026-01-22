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
def test_create_nlp_from_pretraining_config():
    """Test that the default pretraining config validates properly"""
    config = Config().from_str(pretrain_config_string)
    pretrain_config = load_config(DEFAULT_CONFIG_PRETRAIN_PATH)
    filled = config.merge(pretrain_config)
    registry.resolve(filled['pretraining'], schema=ConfigSchemaPretrain)