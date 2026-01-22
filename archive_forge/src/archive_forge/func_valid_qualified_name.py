from collections import defaultdict
import datetime
import io
import json
from prov import Error
from prov.serializers import Serializer
from prov.constants import *
from prov.model import (
import logging
def valid_qualified_name(bundle, value):
    if value is None:
        return None
    qualified_name = bundle.valid_qualified_name(value)
    return qualified_name