import logging
from functools import partial
from .action import ServiceAction
from .action import WaiterAction
from .base import ResourceMeta, ServiceResource
from .collection import CollectionFactory
from .model import ResourceModel
from .response import build_identifiers, ResourceHandler
from ..exceptions import ResourceLoadException
from ..docs import docstring
def property_loader(self):
    if self.meta.data is None:
        if hasattr(self, 'load'):
            self.load()
        else:
            raise ResourceLoadException('{0} has no load method'.format(self.__class__.__name__))
    return self.meta.data.get(name)