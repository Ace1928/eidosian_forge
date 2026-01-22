from saharaclient.api import clusters as cl
from saharaclient.tests.unit import base
from oslo_serialization import jsonutils as json
def test_clusters_delete(self):
    url = self.URL + '/clusters/id'
    self.responses.delete(url, status_code=204)
    self.client.clusters.delete('id')
    self.assertEqual(url, self.responses.last_request.url)