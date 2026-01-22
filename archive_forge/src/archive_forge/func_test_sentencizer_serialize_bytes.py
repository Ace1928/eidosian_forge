import pytest
import spacy
from spacy.lang.en import English
from spacy.pipeline import Sentencizer
from spacy.tokens import Doc
def test_sentencizer_serialize_bytes(en_vocab):
    punct_chars = ['.', '~', '+']
    sentencizer = Sentencizer(punct_chars=punct_chars)
    assert sentencizer.punct_chars == set(punct_chars)
    bytes_data = sentencizer.to_bytes()
    new_sentencizer = Sentencizer(punct_chars=None).from_bytes(bytes_data)
    assert new_sentencizer.punct_chars == set(punct_chars)