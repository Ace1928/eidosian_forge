import openstack.exceptions as exception
from openstack.identity.v3 import (
from openstack.identity.v3 import access_rule as _access_rule
from openstack.identity.v3 import credential as _credential
from openstack.identity.v3 import domain as _domain
from openstack.identity.v3 import domain_config as _domain_config
from openstack.identity.v3 import endpoint as _endpoint
from openstack.identity.v3 import federation_protocol as _federation_protocol
from openstack.identity.v3 import group as _group
from openstack.identity.v3 import identity_provider as _identity_provider
from openstack.identity.v3 import limit as _limit
from openstack.identity.v3 import mapping as _mapping
from openstack.identity.v3 import policy as _policy
from openstack.identity.v3 import project as _project
from openstack.identity.v3 import region as _region
from openstack.identity.v3 import registered_limit as _registered_limit
from openstack.identity.v3 import role as _role
from openstack.identity.v3 import role_assignment as _role_assignment
from openstack.identity.v3 import (
from openstack.identity.v3 import (
from openstack.identity.v3 import (
from openstack.identity.v3 import (
from openstack.identity.v3 import (
from openstack.identity.v3 import (
from openstack.identity.v3 import service as _service
from openstack.identity.v3 import system as _system
from openstack.identity.v3 import trust as _trust
from openstack.identity.v3 import user as _user
from openstack import proxy
from openstack import resource
from openstack import utils
def create_federation_protocol(self, idp_id, **attrs):
    """Create a new federation protocol from attributes

        :param idp_id: The ID of the identity provider or a
            :class:`~openstack.identity.v3.identity_provider.IdentityProvider`
            representing the identity provider the protocol is to be
            attached to.
        :param dict attrs: Keyword arguments which will be used to create a
            :class:`~openstack.identity.v3.federation_protocol.FederationProtocol`,
            comprised of the properties on the
            FederationProtocol class.

        :returns: The results of federation protocol creation
        :rtype:
            :class:`~openstack.identity.v3.federation_protocol.FederationProtocol`
        """
    idp_cls = _identity_provider.IdentityProvider
    if isinstance(idp_id, idp_cls):
        idp_id = idp_id.id
    return self._create(_federation_protocol.FederationProtocol, idp_id=idp_id, **attrs)