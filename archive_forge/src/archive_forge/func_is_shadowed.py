from keyword import iskeyword
import re
from .autocall import IPyAutocall
from traitlets.config.configurable import Configurable
from .inputtransformer2 import (
from .macro import Macro
from .splitinput import LineInfo
from traitlets import (
def is_shadowed(identifier, ip):
    """Is the given identifier defined in one of the namespaces which shadow
    the alias and magic namespaces?  Note that an identifier is different
    than ifun, because it can not contain a '.' character."""
    return identifier in ip.user_ns or identifier in ip.user_global_ns or identifier in ip.ns_table['builtin'] or iskeyword(identifier)