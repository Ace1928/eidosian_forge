import pytest
from spacy.lang.ta import Tamil
@pytest.mark.parametrize('text, num_sents', [(TAMIL_BASIC_TOKENIZER_SENTENCIZER_TEST_TEXT, 9)])
def test_ta_sentencizer(text, num_sents):
    nlp = Tamil()
    nlp.add_pipe('sentencizer')
    doc = nlp(text)
    assert len(list(doc.sents)) == num_sents