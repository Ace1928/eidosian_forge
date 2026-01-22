import collections
import functools
from taskflow import deciders as de
from taskflow import exceptions as exc
from taskflow import flow
from taskflow.types import graph as gr
def iter_links(self):
    return self._get_subgraph().edges(data=True)