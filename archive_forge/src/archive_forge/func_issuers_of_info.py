import logging
from saml2.cache import Cache
def issuers_of_info(self, name_id):
    return self.cache.entities(name_id)