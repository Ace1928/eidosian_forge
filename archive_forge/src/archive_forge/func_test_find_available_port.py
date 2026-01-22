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
def test_find_available_port():
    host = '0.0.0.0'
    port = 5001
    assert find_available_port(port, host) == port, "Port 5001 isn't free"
    from wsgiref.simple_server import demo_app, make_server
    with make_server(host, port, demo_app) as httpd:
        with pytest.warns(UserWarning, match='already in use'):
            found_port = find_available_port(port, host, auto_select=True)
        assert found_port == port + 1, "Didn't find next port"