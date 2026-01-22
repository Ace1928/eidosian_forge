from email.message import Message
from io import BytesIO
from json import dumps, loads
import sys
from wadllib.application import Resource as WadlResource
from lazr.restfulclient import __version__
from lazr.restfulclient._browser import Browser, RestfulHttp
from lazr.restfulclient._json import DatetimeJSONEncoder
from lazr.restfulclient.errors import HTTPError
from lazr.uri import URI
def lp_values_for(self, param_name):
    """Find the set of possible values for a parameter."""
    parameter = self._wadl_resource.get_parameter(param_name, self.JSON_MEDIA_TYPE)
    options = parameter.options
    if len(options) > 0:
        return [option.value for option in options]
    return None