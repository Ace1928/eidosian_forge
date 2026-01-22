import bisect
import re
import traceback
import warnings
from collections import defaultdict, namedtuple
import numpy as np
import param
from packaging.version import Version
from ..core import (
from ..core.ndmapping import item_check
from ..core.operation import Operation
from ..core.options import CallbackError, Cycle
from ..core.spaces import get_nested_streams
from ..core.util import (
from ..element import Points
from ..streams import LinkedStream, Params
from ..util.transform import dim
def undisplayable_info(obj, html=False):
    """Generate helpful message regarding an undisplayable object"""
    collate = '<tt>collate</tt>' if html else 'collate'
    info = 'For more information, please consult the Composing Data tutorial (http://git.io/vtIQh)'
    if isinstance(obj, HoloMap):
        error = f'HoloMap of {obj.type.__name__} objects cannot be displayed.'
        remedy = f'Please call the {collate} method to generate a displayable object'
    elif isinstance(obj, Layout):
        error = 'Layout containing HoloMaps of Layout or GridSpace objects cannot be displayed.'
        remedy = f'Please call the {collate} method on the appropriate elements.'
    elif isinstance(obj, GridSpace):
        error = 'GridSpace containing HoloMaps of Layouts cannot be displayed.'
        remedy = f'Please call the {collate} method on the appropriate elements.'
    if not html:
        return f'{error}\n{remedy}\n{info}'
    else:
        return '<center>{msg}</center>'.format(msg='<br>'.join(['<b>%s</b>' % error, remedy, '<i>%s</i>' % info]))