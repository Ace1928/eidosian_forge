import logging
from saml2.cache import Cache
def get_entityid(self, name_id, source_id, check_not_on_or_after=True):
    try:
        return self.cache.get(name_id, source_id, check_not_on_or_after)['name_id']
    except (KeyError, ValueError):
        return ''