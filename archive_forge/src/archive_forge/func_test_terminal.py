import logging
import sys
import time
import uuid
import pytest
import panel as pn
def test_terminal(document, comm):
    terminal = pn.widgets.Terminal('Hello')
    terminal.write(' World!')
    model = terminal.get_root(document, comm)
    assert model.output == 'Hello World!'
    terminal.clear()
    assert model._clears == 1
    assert terminal.output == ''
    model2 = terminal.get_root(document, comm)
    assert model2.output == ''