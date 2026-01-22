from troveclient import client as base_client
from troveclient.tests import utils
from troveclient.v1 import client
from urllib import parse
def post_clusters(self, body, **kw):
    assert_has_keys(body['cluster'], required=['instances', 'datastore', 'name'])
    if 'instances' in body['cluster']:
        for instance in body['cluster']['instances']:
            assert_has_keys(instance, required=['volume', 'flavorRef'])
    return (202, {}, self.get_clusters_cls_1234()[2])