from oslo_log import log
from keystone.common import manager
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.i18n import _
def get_policy_for_endpoint(self, endpoint_id):

    def _get_policy(policy_id, endpoint_id):
        try:
            return PROVIDERS.policy_api.get_policy(policy_id)
        except exception.PolicyNotFound:
            msg = 'Policy %(policy_id)s referenced in association for endpoint %(endpoint_id)s not found.'
            LOG.warning(msg, {'policy_id': policy_id, 'endpoint_id': endpoint_id})
            raise

    def _look_for_policy_for_region_and_service(endpoint):
        """Look in the region and its parents for a policy.

            Examine the region of the endpoint for a policy appropriate for
            the service of the endpoint. If there isn't a match, then chase up
            the region tree to find one.

            """
        region_id = endpoint['region_id']
        regions_examined = []
        while region_id is not None:
            try:
                ref = self.get_policy_association(service_id=endpoint['service_id'], region_id=region_id)
                return ref['policy_id']
            except exception.PolicyAssociationNotFound:
                pass
            regions_examined.append(region_id)
            region = PROVIDERS.catalog_api.get_region(region_id)
            region_id = None
            if region.get('parent_region_id') is not None:
                region_id = region['parent_region_id']
                if region_id in regions_examined:
                    msg = 'Circular reference or a repeated entry found in region tree - %(region_id)s.'
                    LOG.error(msg, {'region_id': region_id})
                    break
    try:
        ref = self.get_policy_association(endpoint_id=endpoint_id)
        return _get_policy(ref['policy_id'], endpoint_id)
    except exception.PolicyAssociationNotFound:
        pass
    endpoint = PROVIDERS.catalog_api.get_endpoint(endpoint_id)
    policy_id = _look_for_policy_for_region_and_service(endpoint)
    if policy_id is not None:
        return _get_policy(policy_id, endpoint_id)
    try:
        ref = self.get_policy_association(service_id=endpoint['service_id'])
        return _get_policy(ref['policy_id'], endpoint_id)
    except exception.PolicyAssociationNotFound:
        pass
    msg = _('No policy is associated with endpoint %(endpoint_id)s.') % {'endpoint_id': endpoint_id}
    raise exception.NotFound(msg)