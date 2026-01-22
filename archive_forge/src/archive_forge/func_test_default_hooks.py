import pytest
from requests import hooks
def test_default_hooks():
    assert hooks.default_hooks() == {'response': []}