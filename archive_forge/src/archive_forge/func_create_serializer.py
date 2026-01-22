import base64
import calendar
import datetime
import json
import re
from xml.etree import ElementTree
from botocore import validate
from botocore.compat import formatdate
from botocore.exceptions import ParamValidationError
from botocore.utils import (
def create_serializer(protocol_name, include_validation=True):
    serializer = SERIALIZERS[protocol_name]()
    if include_validation:
        validator = validate.ParamValidator()
        serializer = validate.ParamValidationDecorator(validator, serializer)
    return serializer