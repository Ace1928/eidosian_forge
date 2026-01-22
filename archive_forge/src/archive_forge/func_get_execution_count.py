from __future__ import annotations
import builtins
import sys
import typing as t
from IPython.core.displayhook import DisplayHook
from jupyter_client.session import Session, extract_header
from traitlets import Any, Dict, Instance
from ipykernel.jsonutil import encode_images, json_clean
def get_execution_count(self):
    """This method is replaced in kernelapp"""
    return 0