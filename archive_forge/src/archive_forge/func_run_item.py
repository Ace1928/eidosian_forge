from __future__ import absolute_import, division, print_function
import os
import pickle
import tempfile
import warnings
from pytest import Item, hookimpl
from _pytest.runner import runtestprotocol
def run_item(item, nextitem):
    """Run the item in a child process and return a list of reports."""
    with tempfile.NamedTemporaryFile() as temp_file:
        pid = os.fork()
        if not pid:
            temp_file.delete = False
            run_child(item, nextitem, temp_file.name)
        return run_parent(item, pid, temp_file.name)