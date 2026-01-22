import os
from pathlib import Path
import unittest
from traits.api import File, HasTraits, TraitError
from traits.testing.optional_dependencies import requires_traitsui
def test_valid_pathlike_file(self):
    ExampleModel(file_name=Path(__file__))