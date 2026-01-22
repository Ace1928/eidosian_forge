from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.location import FeatureLibLocation
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import byteord, tobytes
from collections import OrderedDict
import itertools
def deviceToString(device):
    if device is None:
        return '<device NULL>'
    else:
        return '<device %s>' % ', '.join(('%d %d' % t for t in device))