from cinderclient.apiclient import base as common_base
from cinderclient import base
def disassociate_all(self, qos_specs):
    """Disassociate all entities from specific qos specs.

        :param qos_specs: The qos specs to be associated with
        """
    resp, body = self.api.client.get('/qos-specs/%s/disassociate_all' % base.getid(qos_specs))
    return common_base.TupleWithMeta((resp, body), resp)