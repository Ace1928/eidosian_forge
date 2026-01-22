from contextlib import contextmanager
from typing import NamedTuple
from functools import partial
from IPython.core.guarded_eval import (
from IPython.testing import decorators as dec
import pytest
def test_rejects_custom_properties():

    class BadProperty:

        @property
        def iloc(self):
            return [None]
    series = BadProperty()
    context = limited(data=series)
    with pytest.raises(GuardRejection):
        guarded_eval('data.iloc[0]', context)