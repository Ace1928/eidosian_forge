import pytest
from hypothesis import settings
from spacy.util import get_lang_class
@pytest.fixture(scope='session')
def zh_tokenizer_pkuseg():
    pytest.importorskip('spacy_pkuseg')
    config = {'nlp': {'tokenizer': {'@tokenizers': 'spacy.zh.ChineseTokenizer', 'segmenter': 'pkuseg'}}, 'initialize': {'tokenizer': {'pkuseg_model': 'web'}}}
    nlp = get_lang_class('zh').from_config(config)
    nlp.initialize()
    return nlp.tokenizer