import logging
from saml2.cache import Cache
def get_info_from(self, name_id, entity_id, check_not_on_or_after=True):
    return self.cache.get(name_id, entity_id, check_not_on_or_after)