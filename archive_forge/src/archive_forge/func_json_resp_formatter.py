from collections import namedtuple
import json
import logging
import pprint
import re
@classmethod
def json_resp_formatter(cls, resp):
    """Override this method to provide custom formatting of json response.
        """
    return json.dumps(resp.value)