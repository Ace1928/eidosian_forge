import inspect
import pytest
from ..widget_string import Combobox, Text
def test_on_submit_deprecation():
    with pytest.deprecated_call() as record:
        Text().on_submit(lambda *args: ...)
    assert len(record) == 1
    assert record[0].filename == inspect.stack(context=0)[1].filename