import suds
from suds import *
import suds.bindings.binding
from suds.builder import Builder
import suds.cache
import suds.metrics as metrics
from suds.options import Options
from suds.plugin import PluginContainer
from suds.properties import Unskin
from suds.reader import DefinitionsReader
from suds.resolver import PathResolver
from suds.sax.document import Document
import suds.sax.parser
from suds.servicedefinition import ServiceDefinition
import suds.transport
import suds.transport.https
from suds.umx.basic import Basic as UmxBasic
from suds.wsdl import Definitions
from . import sudsobject
from http.cookiejar import CookieJar
from copy import deepcopy
import http.client
from logging import getLogger
def last_received(self, d=None):
    """
        Get or set last SOAP received messages document

        To get the last received document call the function without parameter.
        To set the last sent message, pass the document as parameter.

        @param d: A SOAP reply dict message key
        @type string: I{bytes}
        @return: The last received I{soap} message.
        @rtype: L{Document}

        """
    key = 'rx'
    messages = self.client.messages
    if d is None:
        return messages.get(key)
    else:
        messages[key] = d