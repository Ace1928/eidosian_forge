import pytest
from spacy.lang.da.lex_attrs import like_num
def test_da_tokenizer_handles_long_text(da_tokenizer):
    text = 'Der var så dejligt ude på landet. Det var sommer, kornet stod gult, havren grøn,\nhøet var rejst i stakke nede i de grønne enge, og der gik storken på sine lange,\nrøde ben og snakkede ægyptisk, for det sprog havde han lært af sin moder.\n\nRundt om ager og eng var der store skove, og midt i skovene dybe søer; jo, der var rigtignok dejligt derude på landet!'
    tokens = da_tokenizer(text)
    assert len(tokens) == 84