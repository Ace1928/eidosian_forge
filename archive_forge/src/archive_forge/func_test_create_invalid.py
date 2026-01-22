import pytest
from ..base import EngineBase
from ....interfaces import base as nib
from ....interfaces import utility as niu
from ... import engine as pe
@pytest.mark.parametrize('name', ['invalid*1', 'invalid.1', 'invalid@', 'in/valid', None])
def test_create_invalid(name):
    with pytest.raises(ValueError):
        EngineBase(name=name)