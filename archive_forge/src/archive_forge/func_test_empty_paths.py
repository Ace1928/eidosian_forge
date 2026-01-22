from unittest import TestCase
import textwrap
from jsonschema import exceptions
from jsonschema.validators import _LATEST_VERSION
def test_empty_paths(self):
    self.assertShows("\n            Failed validating 'type' in schema:\n                {'type': 'string'}\n\n            On instance:\n                5\n            ", path=[], schema_path=[])