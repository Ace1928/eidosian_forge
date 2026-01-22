from suds import *
from suds.umx import *
from suds.umx.core import Core
from suds.resolver import NodeResolver, Frame
from suds.sudsobject import Factory
from logging import getLogger
def translated(self, value, type):
    """ translate using the schema type """
    if value is not None:
        resolved = type.resolve()
        return resolved.translate(value)
    return value