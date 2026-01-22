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
def test_docbin_user_data_serialized(doc):
    doc.user_data['check'] = True
    nlp = English()
    with make_tempdir() as tmpdir:
        output_file = tmpdir / 'userdata.spacy'
        DocBin(docs=[doc], store_user_data=True).to_disk(output_file)
        reloaded_docs = DocBin().from_disk(output_file).get_docs(nlp.vocab)
        reloaded_doc = list(reloaded_docs)[0]
    assert reloaded_doc.user_data['check'] == True