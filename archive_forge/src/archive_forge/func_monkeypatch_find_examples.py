import os
import pytest
import pyct.cmd
from pyct.cmd import fetch_data, clean_data, copy_examples, examples
@pytest.fixture(autouse=True)
def monkeypatch_find_examples(monkeypatch, tmp_module):
    """Monkeypatching find examples to use a tmp examples.
    """

    def _find_examples(name):
        return os.path.join(str(tmp_module), 'examples')
    monkeypatch.setattr(pyct.cmd, '_find_examples', _find_examples)