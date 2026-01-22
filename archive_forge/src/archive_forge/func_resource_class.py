import abc
import copy
from urllib import parse as urlparse
from ironicclient.common.apiclient import base
from ironicclient import exc
@property
@abc.abstractmethod
def resource_class(self):
    """The resource class

        """