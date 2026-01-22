import logging
from saml2.attribute_resolver import AttributeResolver
from saml2.saml import NAMEID_FORMAT_PERSISTENT
def members_to_ask(self, name_id):
    """Find the member of the Virtual Organization that I haven't already
        spoken too
        """
    vo_members = self._affiliation_members()
    for member in self.member:
        if member not in vo_members:
            vo_members.append(member)
    vo_members = [m for m in vo_members if not self.sp.users.cache.active(name_id, m)]
    logger.info('VO members (not cached): %s', vo_members)
    return vo_members