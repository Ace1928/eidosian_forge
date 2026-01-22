import ctypes
import os
from pathlib import Path
import pytest
from thinc.api import (
from thinc.compat import has_cupy_gpu, has_torch_mps_gpu
from spacy import prefer_gpu, require_cpu, require_gpu, util
from spacy.about import __version__ as spacy_version
from spacy.lang.en import English
from spacy.lang.nl import Dutch
from spacy.language import DEFAULT_CONFIG_PATH
from spacy.ml._precomputable_affine import (
from spacy.schemas import ConfigSchemaTraining, TokenPattern, TokenPatternSchema
from spacy.training.batchers import minibatch_by_words
from spacy.util import (
from .util import get_random_doc, make_tempdir
from spacy import Language
@pytest.mark.issue(6207)
def test_issue6207(en_tokenizer):
    doc = en_tokenizer('zero one two three four five six')
    s1 = doc[:4]
    s2 = doc[3:6]
    s3 = doc[5:7]
    result = util.filter_spans((s1, s2, s3))
    assert s1 in result
    assert s2 not in result
    assert s3 in result