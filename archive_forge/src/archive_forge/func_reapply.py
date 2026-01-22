from troveclient import base
from troveclient import common
from troveclient import utils
def reapply(self, module, md5=None, include_clustered=None, batch_size=None, delay=None, force=None):
    """Reapplies the specified module."""
    url = '/modules/%s/instances' % base.getid(module)
    body = {'reapply': {}}
    if md5:
        body['reapply']['md5'] = md5
    if include_clustered is not None:
        body['reapply']['include_clustered'] = int(include_clustered)
    if batch_size is not None:
        body['reapply']['batch_size'] = batch_size
    if delay is not None:
        body['reapply']['batch_delay'] = delay
    if force is not None:
        body['reapply']['force'] = int(force)
    resp, body = self.api.client.put(url, body=body)
    common.check_for_exceptions(resp, body, url)