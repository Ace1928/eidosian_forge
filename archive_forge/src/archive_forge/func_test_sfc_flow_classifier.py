from openstack.network.v2 import network as _network
from openstack.network.v2 import port as _port
from openstack.network.v2 import sfc_flow_classifier as _flow_classifier
from openstack.network.v2 import subnet as _subnet
from openstack.tests.functional import base
def test_sfc_flow_classifier(self):
    sot = self.operator_cloud.network.find_sfc_flow_classifier(self.FLOW_CLASSIFIER.name)
    self.assertEqual(self.ETHERTYPE, sot.ethertype)
    self.assertEqual(self.SOURCE_IP, sot.source_ip_prefix)
    self.assertEqual(self.PROTOCOL, sot.protocol)
    classifiers = [fc.name for fc in self.operator_cloud.network.sfc_flow_classifiers()]
    self.assertIn(self.FLOW_CLASSIFIER_NAME, classifiers)
    classifier = self.operator_cloud.network.get_sfc_flow_classifier(self.FC_ID)
    self.assertEqual(self.FLOW_CLASSIFIER_NAME, classifier.name)
    self.assertEqual(self.FC_ID, classifier.id)
    classifier = self.operator_cloud.network.update_sfc_flow_classifier(self.FC_ID, name=self.UPDATE_NAME)
    self.assertEqual(self.UPDATE_NAME, classifier.name)