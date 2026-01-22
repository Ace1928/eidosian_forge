import pytest
import pyviz_comms
from pyviz_comms import extension
def test_extension_count_two_cells_parent_first(monkeypatch, get_ipython):
    monkeypatch.setattr(pyviz_comms, 'get_ipython', get_ipython)

    class parent_extension(extension):

        def __call__(self, *args, **params):
            pass

    class sub_extension(parent_extension):

        def __call__(self, *args, **params):
            pass
    parent_extension()
    get_ipython().bump()
    sub_extension()
    assert sub_extension._repeat_execution_in_cell is False
    assert sub_extension._repeat_execution_in_cell == parent_extension._repeat_execution_in_cell
    assert parent_extension._repeat_execution_in_cell == extension._repeat_execution_in_cell
    sub_extension()
    assert sub_extension._repeat_execution_in_cell is True
    assert sub_extension._repeat_execution_in_cell == parent_extension._repeat_execution_in_cell
    assert parent_extension._repeat_execution_in_cell == extension._repeat_execution_in_cell
    parent_extension()
    assert parent_extension._repeat_execution_in_cell is True
    assert parent_extension._repeat_execution_in_cell == sub_extension._repeat_execution_in_cell
    assert sub_extension._repeat_execution_in_cell == extension._repeat_execution_in_cell
    get_ipython().bump()
    parent_extension()
    assert parent_extension._repeat_execution_in_cell is False
    assert parent_extension._repeat_execution_in_cell == sub_extension._repeat_execution_in_cell
    assert sub_extension._repeat_execution_in_cell == extension._repeat_execution_in_cell