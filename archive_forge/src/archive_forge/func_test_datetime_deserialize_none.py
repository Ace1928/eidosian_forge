import pytest
import datetime
import pytz
from traitlets import TraitError
from ..trait_types import (
def test_datetime_deserialize_none():
    assert datetime_from_json(None, None) == None