import contextlib
import re
import tempfile
import numpy
import srsly
from thinc.api import get_current_ops
from spacy.tokens import Doc
from spacy.training import split_bilu_label
from spacy.util import make_tempdir  # noqa: F401
from spacy.vocab import Vocab
@contextlib.contextmanager
def make_tempfile(mode='r'):
    f = tempfile.TemporaryFile(mode=mode)
    yield f
    f.close()