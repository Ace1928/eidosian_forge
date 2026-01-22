import pytest
import spacy
from spacy.lang.en import English
from spacy.tokens import Doc, DocBin
from spacy.tokens.underscore import Underscore
@pytest.mark.issue(4367)
def test_issue4367():
    """Test that docbin init goes well"""
    DocBin()
    DocBin(attrs=['LEMMA'])
    DocBin(attrs=['LEMMA', 'ENT_IOB', 'ENT_TYPE'])