from ipywidgets import register, Widget, DOMWidget, widget_serialization
from traitlets import (
import uuid
from ._version import module_version
def id_gen():
    return uuid.uuid4().urn[9:]