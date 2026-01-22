import numpy
import pytest
from spacy.tokens import Doc, Token
from spacy.vocab import Vocab
def test_doc_retokenize_split_dependencies(en_vocab):
    doc = Doc(en_vocab, words=['LosAngeles', 'start', '.'])
    dep1 = doc.vocab.strings.add('amod')
    dep2 = doc.vocab.strings.add('subject')
    with doc.retokenize() as retokenizer:
        retokenizer.split(doc[0], ['Los', 'Angeles'], [(doc[0], 1), doc[1]], attrs={'dep': [dep1, dep2]})
    assert doc[0].dep == dep1
    assert doc[1].dep == dep2