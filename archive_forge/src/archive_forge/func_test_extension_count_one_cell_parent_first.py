import pytest
import pyviz_comms
from pyviz_comms import extension
def test_extension_count_one_cell_parent_first(monkeypatch, get_ipython):
    monkeypatch.setattr(pyviz_comms, 'get_ipython', get_ipython)

    class parent_extension(extension):

        def __call__(self, *args, **params):
            pass

    class sub_extension(parent_extension):

        def __call__(self, *args, **params):
            pass
    parent_extension()
    assert parent_extension._repeat_execution_in_cell is False
    sub_extension()
    assert sub_extension._repeat_execution_in_cell is True
    parent_extension()
    assert parent_extension._repeat_execution_in_cell is True