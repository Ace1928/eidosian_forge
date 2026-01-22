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
@pytest.mark.parametrize('doc_sizes, expected_batches', [([400, 400, 199], [3]), ([400, 400, 199, 3], [4]), ([400, 400, 199, 3, 200], [3, 2]), ([400, 400, 199, 3, 1], [5]), ([400, 400, 199, 3, 1, 1500], [5]), ([400, 400, 199, 3, 1, 200], [3, 3]), ([400, 400, 199, 3, 1, 999], [3, 3]), ([400, 400, 199, 3, 1, 999, 999], [3, 2, 1, 1]), ([1, 2, 999], [3]), ([1, 2, 999, 1], [4]), ([1, 200, 999, 1], [2, 2]), ([1, 999, 200, 1], [2, 2])])
def test_util_minibatch(doc_sizes, expected_batches):
    docs = [get_random_doc(doc_size) for doc_size in doc_sizes]
    tol = 0.2
    batch_size = 1000
    batches = list(minibatch_by_words(docs, size=batch_size, tolerance=tol, discard_oversize=True))
    assert [len(batch) for batch in batches] == expected_batches
    max_size = batch_size + batch_size * tol
    for batch in batches:
        assert sum([len(doc) for doc in batch]) < max_size