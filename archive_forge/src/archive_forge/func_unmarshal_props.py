import boto
from boto.utils import find_class, Password
from boto.sdb.db.key import Key
from boto.sdb.db.model import Model
from boto.compat import six, encodebytes
from datetime import datetime
from xml.dom.minidom import getDOMImplementation, parse, parseString, Node
def unmarshal_props(self, fp, cls=None, id=None):
    """
        Same as unmarshalling an object, except it returns
        from "get_props_from_doc"
        """
    if isinstance(fp, six.string_types):
        doc = parseString(fp)
    else:
        doc = parse(fp)
    return self.get_props_from_doc(cls, id, doc)