import pytest
import pyviz_comms
from pyviz_comms import extension
def test_extension_count_two_cells_extensions_branched(monkeypatch, get_ipython):
    monkeypatch.setattr(pyviz_comms, 'get_ipython', get_ipython)

    class sub_extension1(extension):

        def __call__(self, *args, **params):
            pass

    class sub_extension2(extension):

        def __call__(self, *args, **params):
            pass
    sub_extension1()
    get_ipython().bump()
    sub_extension2()
    assert sub_extension2._repeat_execution_in_cell is False
    assert sub_extension2._repeat_execution_in_cell == sub_extension1._repeat_execution_in_cell
    assert sub_extension1._repeat_execution_in_cell == extension._repeat_execution_in_cell
    sub_extension2()
    assert sub_extension2._repeat_execution_in_cell is True
    assert sub_extension2._repeat_execution_in_cell == sub_extension1._repeat_execution_in_cell
    assert sub_extension1._repeat_execution_in_cell == extension._repeat_execution_in_cell
    get_ipython().bump()
    sub_extension1()
    assert sub_extension1._repeat_execution_in_cell is False
    assert sub_extension1._repeat_execution_in_cell == sub_extension2._repeat_execution_in_cell
    assert sub_extension2._repeat_execution_in_cell == extension._repeat_execution_in_cell
    get_ipython().bump()
    sub_extension2()
    assert sub_extension2._repeat_execution_in_cell is False
    assert sub_extension2._repeat_execution_in_cell == sub_extension1._repeat_execution_in_cell
    assert sub_extension1._repeat_execution_in_cell == extension._repeat_execution_in_cell