import json
from collections import namedtuple
import macaroonbakery.bakery as bakery
def field(name):
    return serialized.get(name) or serialized.get(name.lower())