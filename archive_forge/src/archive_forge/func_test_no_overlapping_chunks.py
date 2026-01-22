import pytest
from spacy.tokens import Doc
from spacy.util import filter_spans
@pytest.mark.issue(10846)
def test_no_overlapping_chunks(nl_vocab):
    doc = Doc(nl_vocab, words=['Dit', 'programma', 'wordt', 'beschouwd', 'als', "'s", 'werelds', 'eerste', 'computerprogramma'], deps=['det', 'nsubj:pass', 'aux:pass', 'ROOT', 'mark', 'det', 'fixed', 'amod', 'xcomp'], heads=[1, 3, 3, 3, 8, 8, 5, 8, 3], pos=['DET', 'NOUN', 'AUX', 'VERB', 'SCONJ', 'DET', 'NOUN', 'ADJ', 'NOUN'])
    chunks = list(doc.noun_chunks)
    assert filter_spans(chunks) == chunks