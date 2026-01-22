import copy
import testtools
from testtools import matchers
from magnumclient import exceptions
from magnumclient.tests import utils
from magnumclient.v1 import cluster_templates
def test_clustertemplate_show_by_id(self):
    cluster_template = self.mgr.get(CLUSTERTEMPLATE1['id'])
    expect = [('GET', '/v1/clustertemplates/%s' % CLUSTERTEMPLATE1['id'], {}, None)]
    self.assertEqual(expect, self.api.calls)
    self.assertEqual(CLUSTERTEMPLATE1['name'], cluster_template.name)
    self.assertEqual(CLUSTERTEMPLATE1['image_id'], cluster_template.image_id)
    self.assertEqual(CLUSTERTEMPLATE1['docker_volume_size'], cluster_template.docker_volume_size)
    self.assertEqual(CLUSTERTEMPLATE1['docker_storage_driver'], cluster_template.docker_storage_driver)
    self.assertEqual(CLUSTERTEMPLATE1['fixed_network'], cluster_template.fixed_network)
    self.assertEqual(CLUSTERTEMPLATE1['fixed_subnet'], cluster_template.fixed_subnet)
    self.assertEqual(CLUSTERTEMPLATE1['coe'], cluster_template.coe)
    self.assertEqual(CLUSTERTEMPLATE1['http_proxy'], cluster_template.http_proxy)
    self.assertEqual(CLUSTERTEMPLATE1['https_proxy'], cluster_template.https_proxy)
    self.assertEqual(CLUSTERTEMPLATE1['no_proxy'], cluster_template.no_proxy)
    self.assertEqual(CLUSTERTEMPLATE1['network_driver'], cluster_template.network_driver)
    self.assertEqual(CLUSTERTEMPLATE1['volume_driver'], cluster_template.volume_driver)
    self.assertEqual(CLUSTERTEMPLATE1['labels'], cluster_template.labels)
    self.assertEqual(CLUSTERTEMPLATE1['tls_disabled'], cluster_template.tls_disabled)
    self.assertEqual(CLUSTERTEMPLATE1['public'], cluster_template.public)
    self.assertEqual(CLUSTERTEMPLATE1['registry_enabled'], cluster_template.registry_enabled)
    self.assertEqual(CLUSTERTEMPLATE1['master_lb_enabled'], cluster_template.master_lb_enabled)
    self.assertEqual(CLUSTERTEMPLATE1['floating_ip_enabled'], cluster_template.floating_ip_enabled)
    self.assertEqual(CLUSTERTEMPLATE1['hidden'], cluster_template.hidden)