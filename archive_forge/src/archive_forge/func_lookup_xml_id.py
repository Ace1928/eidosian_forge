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
def lookup_xml_id(self, url):
    """A helper method for locating a part of a WADL document.

        :param url: The URL (with anchor) of the desired part of the
        WADL document.
        :return: The XML ID corresponding to the anchor.
        """
    markup_uri = URI(self.markup_url).ensureNoSlash()
    markup_uri.fragment = None
    if url.startswith('http'):
        this_uri = URI(url).ensureNoSlash()
    else:
        this_uri = markup_uri.resolve(url)
    possible_xml_id = this_uri.fragment
    this_uri.fragment = None
    if this_uri == markup_uri:
        return possible_xml_id
    raise NotImplementedError("Can't look up definition in another url (%s)" % url)