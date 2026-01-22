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
def build_request_url(self, param_values=None, **kw_param_values):
    """Return the request URL to use to invoke this method."""
    return self.request.build_url(param_values, **kw_param_values)