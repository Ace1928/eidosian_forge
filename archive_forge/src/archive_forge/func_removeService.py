from zope.interface import Attribute, Interface, implementer
from twisted.internet import defer
from twisted.persisted import sob
from twisted.plugin import IPlugin
from twisted.python import components
from twisted.python.reflect import namedAny
def removeService(self, service):
    if service.name:
        del self.namedServices[service.name]
    self.services.remove(service)
    if self.running:
        return service.stopService()
    else:
        return None