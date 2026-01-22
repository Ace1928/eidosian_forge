import pytest
import spacy
from spacy.lang.en import English
from spacy.pipeline import Sentencizer
from spacy.tokens import Doc
def test_sentencizer_pipe():
    texts = ['Hello! This is a test.', 'Hi! This is a test.']
    nlp = English()
    nlp.add_pipe('sentencizer')
    for doc in nlp.pipe(texts):
        assert doc.has_annotation('SENT_START')
        sent_starts = [t.is_sent_start for t in doc]
        assert sent_starts == [True, False, True, False, False, False, False]
        assert len(list(doc.sents)) == 2
    for ex in nlp.pipe(texts):
        doc = ex.doc
        assert doc.has_annotation('SENT_START')
        sent_starts = [t.is_sent_start for t in doc]
        assert sent_starts == [True, False, True, False, False, False, False]
        assert len(list(doc.sents)) == 2