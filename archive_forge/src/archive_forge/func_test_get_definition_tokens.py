import contextlib
import io
import os
import shutil
import tempfile
import textwrap
import tokenize
import unittest
import unittest.mock as mock
from traits.api import Bool, HasTraits, Int, Property
from traits.testing.optional_dependencies import sphinx, requires_sphinx
def test_get_definition_tokens(self):
    src = textwrap.dedent('        depth_interval = Property(Tuple(Float, Float),\n                                  depends_on="_depth_interval")\n        ')
    string_io = io.StringIO(src)
    tokens = tokenize.generate_tokens(string_io.readline)
    definition_tokens = _get_definition_tokens(tokens)
    string = tokenize.untokenize(definition_tokens)
    self.assertEqual(src.rstrip(), string)