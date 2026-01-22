from ncclient.xml_ import *
from ncclient.operations.errors import OperationError, MissingCapabilityError
def datastore_or_url(wha, loc, capcheck=None):
    node = new_ele(wha)
    if '://' in loc:
        if capcheck is not None:
            capcheck(':url')
            sub_ele(node, 'url').text = loc
    else:
        sub_ele(node, loc)
    return node