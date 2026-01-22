from saharaclient.api import plugins
from saharaclient.tests.unit import base
def test_convert_to_cluster_template(self):
    url = self.URL + '/plugins/plugin/1/convert-config/template'
    response = {'name': 'name', 'description': 'description', 'plugin_name': 'plugin', 'hadoop_version': '1'}
    self.responses.post(url, status_code=202, json={'cluster_template': response})
    resp = self.client.plugins.convert_to_cluster_template('plugin', 1, 'template', 'file')
    self.assertEqual(url, self.responses.last_request.url)
    self.assertEqual(response, resp)