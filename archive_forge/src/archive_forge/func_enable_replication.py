from cinderclient import api_versions
from cinderclient.apiclient import base as common_base
from cinderclient import base
from cinderclient import utils
@api_versions.wraps('3.38')
def enable_replication(self, group):
    """Enables replication for a group.

        :param group: the :class:`Group` to enable replication.
        """
    body = {'enable_replication': {}}
    self.run_hooks('modify_body_for_action', body, 'group')
    url = '/groups/%s/action' % base.getid(group)
    resp, body = self.api.client.post(url, body=body)
    return common_base.TupleWithMeta((resp, body), resp)