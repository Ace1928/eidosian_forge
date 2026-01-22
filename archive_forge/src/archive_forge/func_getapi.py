import logging
import sys
import weakref
import webob
from wsme.exc import ClientSideError, UnknownFunction
from wsme.protocol import getprotocol
from wsme.rest import scan_api
import wsme.api
import wsme.types
def getapi(self):
    """
        Returns the api description.

        :rtype: list of (path, :class:`FunctionDefinition`)
        """
    if self._api is None:
        self._api = [(path, f, f._wsme_definition, args) for path, f, args in self._scan_api(self)]
        for path, f, fdef, args in self._api:
            fdef.resolve_types(self.__registry__)
    return [(path, fdef) for path, f, fdef, args in self._api]