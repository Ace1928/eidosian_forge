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
@pytest.mark.issue(2800)
def test_issue2800():
    """Test issue that arises when too many labels are added to NER model.
    Used to cause segfault.
    """
    nlp = English()
    train_data = []
    train_data.extend([Example.from_dict(nlp.make_doc('One sentence'), {'entities': []})])
    entity_types = [str(i) for i in range(1000)]
    ner = nlp.add_pipe('ner')
    for entity_type in list(entity_types):
        ner.add_label(entity_type)
    optimizer = nlp.initialize()
    for i in range(20):
        losses = {}
        random.shuffle(train_data)
        for example in train_data:
            nlp.update([example], sgd=optimizer, losses=losses, drop=0.5)