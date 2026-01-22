import traceback
import pytest
from wasabi.traceback_printer import TracebackPrinter
def test_traceback_tb_base_none(tb):
    tbp = TracebackPrinter()
    msg = tbp('Hello world', tb=tb)
    print(msg)