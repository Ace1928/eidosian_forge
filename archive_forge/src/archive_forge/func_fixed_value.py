import datetime
from email.utils import quote
import io
import json
import random
import re
import sys
import time
from lazr.uri import URI, merge
from wadllib import (
from wadllib.iso_strptime import iso_strptime
@property
def fixed_value(self):
    """The value to which this parameter is fixed, if any.

        A fixed parameter must be present in invocations of a WADL
        method, and it must have a particular value. This is commonly
        used to designate one parameter as containing the name of the
        server-side operation to be invoked.
        """
    return self.tag.attrib.get('fixed')