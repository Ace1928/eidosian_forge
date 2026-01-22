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
def test_train_negative_deprecated():
    """Test that the deprecated negative entity format raises a custom error."""
    train_data = [('Who is Shaka Khan?', {'entities': [(7, 17, '!PERSON')]})]
    nlp = English()
    train_examples = []
    for t in train_data:
        train_examples.append(Example.from_dict(nlp.make_doc(t[0]), t[1]))
    ner = nlp.add_pipe('ner', last=True)
    ner.add_label('PERSON')
    nlp.initialize()
    for itn in range(2):
        losses = {}
        batches = util.minibatch(train_examples, size=8)
        for batch in batches:
            with pytest.raises(ValueError):
                nlp.update(batch, losses=losses)