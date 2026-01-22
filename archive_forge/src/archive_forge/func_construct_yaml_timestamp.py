from __future__ import print_function, absolute_import, division
import datetime
import base64
import binascii
import re
import sys
import types
import warnings
from ruamel.yaml.error import (MarkedYAMLError, MarkedYAMLFutureWarning,
from ruamel.yaml.nodes import *                               # NOQA
from ruamel.yaml.nodes import (SequenceNode, MappingNode, ScalarNode)
from ruamel.yaml.compat import (utf8, builtins_module, to_str, PY2, PY3,  # NOQA
from ruamel.yaml.comments import *                               # NOQA
from ruamel.yaml.comments import (CommentedMap, CommentedOrderedMap, CommentedSet,
from ruamel.yaml.scalarstring import (SingleQuotedScalarString, DoubleQuotedScalarString,
from ruamel.yaml.scalarint import ScalarInt, BinaryInt, OctalInt, HexInt, HexCapsInt
from ruamel.yaml.scalarfloat import ScalarFloat
from ruamel.yaml.scalarbool import ScalarBoolean
from ruamel.yaml.timestamp import TimeStamp
from ruamel.yaml.util import RegExp
def construct_yaml_timestamp(self, node, values=None):
    try:
        match = self.timestamp_regexp.match(node.value)
    except TypeError:
        match = None
    if match is None:
        raise ConstructorError(None, None, 'failed to construct timestamp from "{}"'.format(node.value), node.start_mark)
    values = match.groupdict()
    if not values['hour']:
        return SafeConstructor.construct_yaml_timestamp(self, node, values)
    for part in ['t', 'tz_sign', 'tz_hour', 'tz_minute']:
        if values[part]:
            break
    else:
        return SafeConstructor.construct_yaml_timestamp(self, node, values)
    year = int(values['year'])
    month = int(values['month'])
    day = int(values['day'])
    hour = int(values['hour'])
    minute = int(values['minute'])
    second = int(values['second'])
    fraction = 0
    if values['fraction']:
        fraction_s = values['fraction'][:6]
        while len(fraction_s) < 6:
            fraction_s += '0'
        fraction = int(fraction_s)
        if len(values['fraction']) > 6 and int(values['fraction'][6]) > 4:
            fraction += 1
    delta = None
    if values['tz_sign']:
        tz_hour = int(values['tz_hour'])
        minutes = values['tz_minute']
        tz_minute = int(minutes) if minutes else 0
        delta = datetime.timedelta(hours=tz_hour, minutes=tz_minute)
        if values['tz_sign'] == '-':
            delta = -delta
    if delta:
        dt = datetime.datetime(year, month, day, hour, minute)
        dt -= delta
        data = TimeStamp(dt.year, dt.month, dt.day, dt.hour, dt.minute, second, fraction)
        data._yaml['delta'] = delta
        tz = values['tz_sign'] + values['tz_hour']
        if values['tz_minute']:
            tz += ':' + values['tz_minute']
        data._yaml['tz'] = tz
    else:
        data = TimeStamp(year, month, day, hour, minute, second, fraction)
        if values['tz']:
            data._yaml['tz'] = values['tz']
    if values['t']:
        data._yaml['t'] = True
    return data