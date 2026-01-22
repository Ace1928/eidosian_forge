import sys
import uuid
import ovs.db.data
import ovs.db.parser
import ovs.ovsuuid
from ovs.db import error
def commafy(x):
    """Returns integer x formatted in decimal with thousands set off by
    commas."""
    return _commafy('%d' % x)