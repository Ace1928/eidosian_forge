import base64
import http.client
import json
import logging
import re
from botocore.compat import ETree, XMLParseError
from botocore.eventstream import EventStream, NoInitialResponseError
from botocore.utils import (
def set_parser_defaults(self, **kwargs):
    """Set default arguments when a parser instance is created.

        You can specify any kwargs that are allowed by a ResponseParser
        class.  There are currently two arguments:

            * timestamp_parser - A callable that can parse a timestamp string
            * blob_parser - A callable that can parse a blob type

        """
    self._defaults.update(kwargs)