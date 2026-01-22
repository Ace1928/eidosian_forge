import pytest
from unittest import mock
from traitlets import Bool, Tuple, List, Instance, CFloat, CInt, Float, Int, TraitError, observe
from .utils import setup, teardown
import ipywidgets
from ipywidgets import Widget
def transform_fromjson(data, widget):
    if not data[0]:
        return data
    return [False] + data[1:-2] + [data[-1], data[-2]]