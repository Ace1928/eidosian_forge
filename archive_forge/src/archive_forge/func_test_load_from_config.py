import pytest
from .util import load_logger_from_config
def test_load_from_config():
    valid_logger, nlp = load_logger_from_config(valid_config_string)
    valid_logger(nlp)
    with pytest.raises(ValueError, match='No loggers'):
        invalid_logger, nlp = load_logger_from_config(invalid_config_string)
        invalid_logger(nlp)