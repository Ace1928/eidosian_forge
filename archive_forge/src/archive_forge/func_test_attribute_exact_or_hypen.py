import pytest
import types
from unittest.mock import MagicMock
from bs4 import (
from . import (
def test_attribute_exact_or_hypen(self):
    self.assert_select_multiple(('p[lang|="en"]', ['lang-en', 'lang-en-gb', 'lang-en-us']), ('[lang|="en"]', ['lang-en', 'lang-en-gb', 'lang-en-us']), ('p[lang|="fr"]', ['lang-fr']), ('p[lang|="gb"]', []))