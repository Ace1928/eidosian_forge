import logging
import random
import pytest
from numpy.testing import assert_equal
from spacy import registry, util
from spacy.attrs import ENT_IOB
from spacy.lang.en import English
from spacy.lang.it import Italian
from spacy.language import Language
from spacy.lookups import Lookups
from spacy.pipeline import EntityRecognizer
from spacy.pipeline._parser_internals.ner import BiluoPushDown
from spacy.pipeline.ner import DEFAULT_NER_MODEL
from spacy.tokens import Doc, Span
from spacy.training import Example, iob_to_biluo, split_bilu_label
from spacy.vocab import Vocab
from ..util import make_tempdir
def test_beam_ner_scores():
    beam_width = 16
    beam_density = 0.0001
    nlp = English()
    config = {'beam_width': beam_width, 'beam_density': beam_density}
    ner = nlp.add_pipe('beam_ner', config=config)
    train_examples = []
    for text, annotations in TRAIN_DATA:
        train_examples.append(Example.from_dict(nlp.make_doc(text), annotations))
        for ent in annotations.get('entities'):
            ner.add_label(ent[2])
    optimizer = nlp.initialize()
    losses = {}
    nlp.update(train_examples, sgd=optimizer, losses=losses)
    test_text = 'I like London.'
    doc = nlp.make_doc(test_text)
    docs = [doc]
    beams = ner.predict(docs)
    entity_scores = ner.scored_ents(beams)[0]
    for j in range(len(doc)):
        for label in ner.labels:
            score = entity_scores[j, j + 1, label]
            eps = 1e-05
            assert 0 - eps <= score <= 1 + eps