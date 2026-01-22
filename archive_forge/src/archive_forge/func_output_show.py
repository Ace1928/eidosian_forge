from urllib import parse
from heatclient._i18n import _
from heatclient.common import base
from heatclient.common import utils
from heatclient import exc
def output_show(self, stack_id, output_key):
    stack_identifier = self._resolve_stack_id(stack_id)
    resp = self.client.get('/stacks/%(id)s/outputs/%(key)s' % {'id': stack_identifier, 'key': output_key})
    body = utils.get_response_body(resp)
    return body