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
def test_training_before_update(doc):

    def before_update(nlp, args):
        assert args['step'] == 0
        assert args['epoch'] == 1
        raise ValueError('ran_before_update')

    def generate_batch():
        yield (1, [Example(doc, doc)])
    nlp = spacy.blank('en')
    nlp.add_pipe('tagger')
    optimizer = Adam()
    generator = train_while_improving(nlp, optimizer, generate_batch(), lambda: None, dropout=0.1, eval_frequency=100, accumulate_gradient=10, patience=10, max_steps=100, exclude=[], annotating_components=[], before_update=before_update)
    with pytest.raises(ValueError, match='ran_before_update'):
        for _ in generator:
            pass