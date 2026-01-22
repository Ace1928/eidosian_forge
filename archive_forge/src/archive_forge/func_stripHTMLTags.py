import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import collections
import pprint
import traceback
import types
from datetime import datetime
@staticmethod
def stripHTMLTags(s, l, tokens):
    """
        Parse action to remove HTML tags from web page HTML source

        Example::
            # strip HTML links from normal text 
            text = '<td>More info at the <a href="http://pyparsing.wikispaces.com">pyparsing</a> wiki page</td>'
            td,td_end = makeHTMLTags("TD")
            table_text = td + SkipTo(td_end).setParseAction(pyparsing_common.stripHTMLTags)("body") + td_end
            
            print(table_text.parseString(text).body) # -> 'More info at the pyparsing wiki page'
        """
    return pyparsing_common._html_stripper.transformString(tokens[0])