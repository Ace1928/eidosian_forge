import logging
from saml2.cache import Cache
def remove_person(self, name_id):
    self.cache.delete(name_id)