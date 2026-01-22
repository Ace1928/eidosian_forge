import pytest
from hypothesis import settings
from spacy.util import get_lang_class
@pytest.fixture(scope='session')
def zh_tokenizer_jieba():
    pytest.importorskip('jieba')
    config = {'nlp': {'tokenizer': {'@tokenizers': 'spacy.zh.ChineseTokenizer', 'segmenter': 'jieba'}}}
    nlp = get_lang_class('zh').from_config(config)
    return nlp.tokenizer