import random
import numpy
import pytest
import srsly
from thinc.api import Adam, compounding
import spacy
from spacy.lang.en import English
from spacy.tokens import Doc, DocBin
from spacy.training import (
from spacy.training.align import get_alignments
from spacy.training.alignment_array import AlignmentArray
from spacy.training.converters import json_to_docs
from spacy.training.loop import train_while_improving
from spacy.util import (
from ..util import make_tempdir
@pytest.mark.filterwarnings('ignore::UserWarning')
def test_json_to_docs_no_ner(en_vocab):
    data = [{'id': 1, 'paragraphs': [{'sentences': [{'tokens': [{'dep': 'nn', 'head': 1, 'tag': 'NNP', 'orth': 'Ms.'}, {'dep': 'nsubj', 'head': 1, 'tag': 'NNP', 'orth': 'Haag'}, {'dep': 'ROOT', 'head': 0, 'tag': 'VBZ', 'orth': 'plays'}, {'dep': 'dobj', 'head': -1, 'tag': 'NNP', 'orth': 'Elianti'}, {'dep': 'punct', 'head': -2, 'tag': '.', 'orth': '.'}]}]}]}]
    docs = list(json_to_docs(data))
    assert len(docs) == 1
    for doc in docs:
        assert not doc.has_annotation('ENT_IOB')
    for token in doc:
        assert token.ent_iob == 0
    eg = Example(Doc(doc.vocab, words=[w.text for w in doc], spaces=[bool(w.whitespace_) for w in doc]), doc)
    ner_tags = eg.get_aligned_ner()
    assert ner_tags == [None, None, None, None, None]