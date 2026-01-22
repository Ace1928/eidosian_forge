import pytest
from spacy.language import Language
from spacy.pipeline.functions import merge_subtokens
from spacy.tokens import Doc, Span
from ..doc.test_underscore import clean_underscore  # noqa: F401
def test_token_splitter():
    nlp = Language()
    config = {'min_length': 20, 'split_length': 5}
    token_splitter = nlp.add_pipe('token_splitter', config=config)
    doc = nlp('aaaaabbbbbcccccdddd e f g')
    assert [t.text for t in doc] == ['aaaaabbbbbcccccdddd', 'e', 'f', 'g']
    doc = nlp('aaaaabbbbbcccccdddddeeeeeff g h i')
    assert [t.text for t in doc] == ['aaaaa', 'bbbbb', 'ccccc', 'ddddd', 'eeeee', 'ff', 'g', 'h', 'i']
    assert all((len(t.text) <= token_splitter.split_length for t in doc))