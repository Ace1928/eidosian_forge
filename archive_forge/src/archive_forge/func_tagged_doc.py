import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal
from pytest import approx
from spacy.lang.en import English
from spacy.scorer import PRFScore, ROCAUCScore, Scorer, _roc_auc_score, _roc_curve
from spacy.tokens import Doc, Span
from spacy.training import Example
from spacy.training.iob_utils import offsets_to_biluo_tags
@pytest.fixture
def tagged_doc():
    text = "Sarah's sister flew to Silicon Valley via London."
    tags = ['NNP', 'POS', 'NN', 'VBD', 'IN', 'NNP', 'NNP', 'IN', 'NNP', '.']
    pos = ['PROPN', 'PART', 'NOUN', 'VERB', 'ADP', 'PROPN', 'PROPN', 'ADP', 'PROPN', 'PUNCT']
    morphs = ['NounType=prop|Number=sing', 'Poss=yes', 'Number=sing', 'Tense=past|VerbForm=fin', '', 'NounType=prop|Number=sing', 'NounType=prop|Number=sing', '', 'NounType=prop|Number=sing', 'PunctType=peri']
    nlp = English()
    doc = nlp(text)
    for i in range(len(tags)):
        doc[i].tag_ = tags[i]
        doc[i].pos_ = pos[i]
        doc[i].set_morph(morphs[i])
        if i > 0:
            doc[i].is_sent_start = False
    return doc