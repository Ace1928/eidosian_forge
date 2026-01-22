from suds import *
from suds.umx import *
from suds.umx.core import Core
from suds.resolver import NodeResolver, Frame
from suds.sudsobject import Factory
from logging import getLogger
def nillable(self, content):
    resolved = content.type.resolve()
    return content.type.nillable or (resolved.builtin() and resolved.nillable)