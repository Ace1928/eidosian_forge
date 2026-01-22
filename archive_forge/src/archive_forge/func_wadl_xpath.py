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
def wadl_xpath(tag_name):
    """Turn a tag name into an XPath path."""
    return './' + wadl_tag(tag_name)