import enum
import unittest
from traits.api import (
from traits.etsconfig.api import ETSConfig
from traits.testing.optional_dependencies import requires_traitsui
def test_valid_enum(self):
    example_model = ExampleModel(root='model1')
    example_model.root = 'model2'