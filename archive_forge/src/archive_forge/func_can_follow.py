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
def can_follow(self):
    """Can this link be followed within wadllib?

        wadllib can follow a link if it points to a resource that has
        a WADL definition.
        """
    try:
        definition_url = self._get_definition_url()
    except WADLError:
        return False
    return True