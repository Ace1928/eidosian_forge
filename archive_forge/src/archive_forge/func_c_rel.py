import logging
from os_ken.services.protocols.bgp.operator.views import fields
def c_rel(self, *args, **kwargs):
    return self.combine_related(*args, **kwargs)