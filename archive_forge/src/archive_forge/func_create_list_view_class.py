import logging
from os_ken.services.protocols.bgp.operator.views import fields
def create_list_view_class(detail_view_class, name):
    encode = None
    if 'encode' in dir(detail_view_class):

        def encode(self):
            return RdyToFlattenList([detail_view_class(obj).encode() for obj in self.model])
    return _create_collection_view(detail_view_class, name, encode, OperatorListView)