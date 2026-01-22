import argparse
from unittest import mock
from osc_lib.tests import utils
from oslo_utils import uuidutils
from openstack.network.v2 import sfc_flow_classifier as flow_classifier
from openstack.network.v2 import sfc_port_chain as port_chain
from openstack.network.v2 import sfc_port_pair as port_pair
from openstack.network.v2 import sfc_port_pair_group as port_pair_group
from openstack.network.v2 import sfc_service_graph as service_graph
@staticmethod
def create_sfc_service_graphs(attrs=None, count=1):
    """Create multiple service graphs.

        :param Dictionary attrs:
            A dictionary with all attributes
        :param int count:
            The number of service graphs to fake
        :return:
            A list of dictionaries faking the service graphs.
        """
    service_graphs = []
    for _ in range(count):
        service_graphs.append(FakeSfcServiceGraph.create_sfc_service_graph(attrs))
    return service_graphs