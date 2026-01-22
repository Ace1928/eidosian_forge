from ipywidgets import Widget, widget_serialization
from traitlets import Unicode
from .._package import npm_pkg_name
from .._version import EXTENSION_SPEC_VERSION
def exec_three_obj_method(self, method_name, *args, **kwargs):
    """Execute a method on the three object.

        Excute the method specified by `method_name` on the three
        object, with arguments `args`. `kwargs` is currently ignored.
        """
    content = {'type': 'exec_three_obj_method', 'method_name': method_name, 'args': widget_serialization['to_json'](args, None)}
    self.send(content=content, buffers=None)