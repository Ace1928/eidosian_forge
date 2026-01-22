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
@pytest.mark.parametrize('parser_config_string', [parser_config_string_upper, parser_config_string_no_upper])
def test_config_validate_literal(parser_config_string):
    nlp = English()
    config = Config().from_str(parser_config_string)
    config['model']['state_type'] = 'nonsense'
    with pytest.raises(ConfigValidationError):
        nlp.add_pipe('parser', config=config)
    config['model']['state_type'] = 'ner'
    nlp.add_pipe('parser', config=config)