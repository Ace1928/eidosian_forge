from html.entities import codepoint2name
from collections import defaultdict
import codecs
import re
import logging
import string
from html.entities import html5
@classmethod
def quoted_attribute_value(self, value):
    """Make a value into a quoted XML attribute, possibly escaping it.

         Most strings will be quoted using double quotes.

          Bob's Bar -> "Bob's Bar"

         If a string contains double quotes, it will be quoted using
         single quotes.

          Welcome to "my bar" -> 'Welcome to "my bar"'

         If a string contains both single and double quotes, the
         double quotes will be escaped, and the string will be quoted
         using double quotes.

          Welcome to "Bob's Bar" -> "Welcome to &quot;Bob's bar&quot;
        """
    quote_with = '"'
    if '"' in value:
        if "'" in value:
            replace_with = '&quot;'
            value = value.replace('"', replace_with)
        else:
            quote_with = "'"
    return quote_with + value + quote_with