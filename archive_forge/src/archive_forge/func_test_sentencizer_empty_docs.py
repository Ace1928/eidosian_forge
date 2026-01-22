import pytest
import spacy
from spacy.lang.en import English
from spacy.pipeline import Sentencizer
from spacy.tokens import Doc
def test_sentencizer_empty_docs():
    one_empty_text = ['']
    many_empty_texts = ['', '', '']
    some_empty_texts = ['hi', '', 'This is a test. Here are two sentences.', '']
    nlp = English()
    nlp.add_pipe('sentencizer')
    for texts in [one_empty_text, many_empty_texts, some_empty_texts]:
        for doc in nlp.pipe(texts):
            assert doc.has_annotation('SENT_START')
            sent_starts = [t.is_sent_start for t in doc]
            if len(doc) == 0:
                assert sent_starts == []
            else:
                assert len(sent_starts) > 0