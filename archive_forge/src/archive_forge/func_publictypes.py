from suds import *
import suds.metrics as metrics
from suds.sax import Namespace
from logging import getLogger
def publictypes(self):
    """Get all public types."""
    for t in list(self.wsdl.schema.types.values()):
        if t in self.params:
            continue
        if t in self.types:
            continue
        item = (t, t)
        self.types.append(item)
    self.types.sort(key=lambda x: x[0].name)