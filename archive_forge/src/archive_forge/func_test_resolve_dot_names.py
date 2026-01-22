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
def test_resolve_dot_names():
    config = {'training': {'optimizer': {'@optimizers': 'Adam.v1'}}, 'foo': {'bar': 'training.optimizer', 'baz': 'training.xyz'}}
    result = util.resolve_dot_names(config, ['training.optimizer'])
    assert isinstance(result[0], Optimizer)
    with pytest.raises(ConfigValidationError) as e:
        util.resolve_dot_names(config, ['training.xyz', 'training.optimizer'])
    errors = e.value.errors
    assert len(errors) == 1
    assert errors[0]['loc'] == ['training', 'xyz']