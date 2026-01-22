import pytest
from thinc.api import ConfigValidationError
from spacy.lang.zh import Chinese, _get_pkuseg_trie_data
def test_zh_unsupported_segmenter():
    config = {'nlp': {'tokenizer': {'segmenter': 'unk'}}}
    with pytest.raises(ConfigValidationError):
        Chinese.from_config(config)