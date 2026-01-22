from __future__ import absolute_import, division, print_function
import json
import hashlib
def marshal(data, keys):
    ordered = OrderedDict()
    for key in keys:
        ordered[key] = data.get(key, '')
    return json.dumps(ordered, separators=(',', ':')).encode('utf-8')