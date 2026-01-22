import json
import os
import sys
import uuid
from collections import defaultdict
from contextlib import contextmanager
from itertools import product
import param
from bokeh.core.property.bases import Property
from bokeh.models import CustomJS
from param.parameterized import Watcher
from ..util import param_watchers
from .model import add_to_doc, diff
from .state import state
def link_to_jslink(model, source, src_spec, target, tgt_spec):
    """
    Converts links declared in Python into JS Links by using the
    declared forward and reverse JS transforms on the source and target.
    """
    ref = model.ref['id']
    if source._source_transforms.get(src_spec, False) is None or target._target_transforms.get(tgt_spec, False) is None or ref not in source._models or (ref not in target._models):
        return
    from ..links import JSLinkCallbackGenerator, Link
    properties = dict(value=target._rename.get(tgt_spec, tgt_spec))
    link = Link(source, target, bidirectional=True, properties=properties)
    JSLinkCallbackGenerator(model, link, source, target)
    return link