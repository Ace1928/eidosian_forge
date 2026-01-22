import pytest
import pyviz_comms
from pyviz_comms import extension
def test_get_ipython_fixture(monkeypatch, get_ipython):
    monkeypatch.setattr(pyviz_comms, 'get_ipython', get_ipython)

    class sub_extension(extension):

        def __call__(self, *args, **params):
            pass
    sub_extension()
    assert sub_extension._last_execution_count == 1
    sub_extension()
    assert sub_extension._last_execution_count == 1
    get_ipython().bump()
    sub_extension()
    assert sub_extension._last_execution_count == 2