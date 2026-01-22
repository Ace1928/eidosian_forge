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
@registry.architectures('my_test_parser')
def my_parser():
    tok2vec = build_Tok2Vec_model(MultiHashEmbed(width=321, attrs=['LOWER', 'SHAPE'], rows=[5432, 5432], include_static_vectors=False), MaxoutWindowEncoder(width=321, window_size=3, maxout_pieces=4, depth=2))
    parser = build_tb_parser_model(tok2vec=tok2vec, state_type='parser', extra_state_tokens=True, hidden_width=65, maxout_pieces=5, use_upper=True)
    return parser