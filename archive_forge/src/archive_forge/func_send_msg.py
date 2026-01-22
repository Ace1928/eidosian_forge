from ipywidgets import DOMWidget, Widget, widget_serialization
from traitlets import Unicode, CInt, Enum, Bool, Int, Instance, List
from .._package import npm_pkg_name
from .._version import EXTENSION_SPEC_VERSION
from .Three import ThreeWidget
from ..enums import ToneMappings
from ..math.Plane_autogen import Plane
from ..renderers.webgl.WebGLShadowMap_autogen import WebGLShadowMap
from ..traits import IEEEFloat
def send_msg(self, message_type, payload=None):
    if payload is None:
        payload = {}
    content = {'type': message_type, 'payload': payload}
    self.send(content=content, buffers=None)