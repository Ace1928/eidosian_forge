from saharaclient.api import clusters as cl
from saharaclient.tests.unit import base
from oslo_serialization import jsonutils as json
def test_clusters_get_with_progress(self):
    url = self.URL + '/clusters/id?show_progress=True'
    self.responses.get(url, json={'cluster': self.body_with_progress})
    resp = self.client.clusters.get('id', show_progress=True)
    self.assertEqual(url, self.responses.last_request.url)
    self.assertIsInstance(resp, cl.Cluster)
    self.assertFields(self.body, resp)