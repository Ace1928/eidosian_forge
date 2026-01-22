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
def process_reply(self, reply, status, description):
    """
        Process a web service operation SOAP reply.

        Depending on how the ``retxml`` option is set, may return the SOAP
        reply XML or process it and return the Python object representing the
        returned value.

        @param reply: The SOAP reply envelope.
        @type reply: I{bytes}
        @param status: The HTTP status code (None indicates httplib.OK).
        @type status: int|I{None}
        @param description: Additional status description.
        @type description: str
        @return: The invoked web service operation return value.
        @rtype: I{builtin}|I{subclass of} L{Object}|I{bytes}|I{None}

        """
    if status is None:
        status = http.client.OK
    debug_message = 'Reply HTTP status - %d' % (status,)
    if status in (http.client.ACCEPTED, http.client.NO_CONTENT):
        log.debug(debug_message)
        return
    if status == http.client.OK:
        log.debug('%s\n%s', debug_message, reply)
    else:
        log.debug('%s - %s\n%s', debug_message, description, reply)
    plugins = PluginContainer(self.options.plugins)
    ctx = plugins.message.received(reply=reply)
    reply = ctx.reply
    replyroot = None
    if status in (http.client.OK, http.client.INTERNAL_SERVER_ERROR):
        replyroot = _parse(reply)
        if len(reply) > 0:
            self.last_received(replyroot)
        plugins.message.parsed(reply=replyroot)
        fault = self.__get_fault(replyroot)
        if fault:
            if status != http.client.INTERNAL_SERVER_ERROR:
                log.warning('Web service reported a SOAP processing fault using an unexpected HTTP status code %d. Reporting as an internal server error.', status)
            if self.options.faults:
                raise WebFault(fault, replyroot)
            return (http.client.INTERNAL_SERVER_ERROR, fault)
    if status != http.client.OK:
        if self.options.faults:
            raise Exception((status, description))
        return (status, description)
    if self.options.retxml:
        return reply
    result = replyroot and self.method.binding.output.get_reply(self.method, replyroot)
    ctx = plugins.message.unmarshalled(reply=result)
    result = ctx.reply
    if self.options.faults:
        return result
    return (http.client.OK, result)