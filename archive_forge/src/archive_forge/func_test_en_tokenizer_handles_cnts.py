import pytest
from spacy.lang.en.lex_attrs import like_num
@pytest.mark.parametrize('text,length', [('The U.S. Army likes Shock and Awe.', 8), ('U.N. regulations are not a part of their concern.', 10), ("“Isn't it?”", 6), ('Yes! "I\'d rather have a walk", Ms. Comble sighed. ', 15), ("'Me too!', Mr. P. Delaware cried. ", 11), ('They ran about 10km.', 6), ('But then the 6,000-year ice age came...', 10)])
def test_en_tokenizer_handles_cnts(en_tokenizer, text, length):
    tokens = en_tokenizer(text)
    assert len(tokens) == length