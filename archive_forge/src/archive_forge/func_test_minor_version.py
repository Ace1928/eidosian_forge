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
@pytest.mark.parametrize('a1,a2,b1,b2,is_match', [('3.0.0', '3.0', '3.0.1', '3.0', True), ('3.1.0', '3.1', '3.2.1', '3.2', False), ('xxx', None, '1.2.3.dev0', '1.2', False)])
def test_minor_version(a1, a2, b1, b2, is_match):
    assert util.get_minor_version(a1) == a2
    assert util.get_minor_version(b1) == b2
    assert util.is_minor_version_match(a1, b1) is is_match
    assert util.is_minor_version_match(a2, b2) is is_match