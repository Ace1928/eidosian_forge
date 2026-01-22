from urllib import parse
from heatclient._i18n import _
from heatclient.common import base
from heatclient.common import utils
from heatclient import exc
def snapshot_show(self, stack_id, snapshot_id):
    stack_identifier = self._resolve_stack_id(stack_id)
    resp = self.client.get('/stacks/%s/snapshots/%s' % (stack_identifier, snapshot_id))
    body = utils.get_response_body(resp)
    return body