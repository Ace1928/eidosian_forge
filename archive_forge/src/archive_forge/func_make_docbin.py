import random
from contextlib import contextmanager
import pytest
from spacy.lang.en import English
from spacy.pipeline._parser_internals.nonproj import contains_cycle
from spacy.tokens import Doc, DocBin, Span
from spacy.training import Corpus, Example
from spacy.training.augment import (
from ..util import make_tempdir
@contextmanager
def make_docbin(docs, name='roundtrip.spacy'):
    with make_tempdir() as tmpdir:
        output_file = tmpdir / name
        DocBin(docs=docs).to_disk(output_file)
        yield output_file