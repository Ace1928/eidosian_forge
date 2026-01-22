import datetime
from apitools.gen import message_registry
from apitools.gen import service_registry
from apitools.gen import util
@property
def discovery_doc(self):
    return self.__discovery_doc