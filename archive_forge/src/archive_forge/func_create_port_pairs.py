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
def create_port_pairs(attrs=None, count=1):
    """Create multiple port_pairs.

        :param Dictionary attrs:
            A dictionary with all attributes
        :param int count:
            The number of port_pairs to fake
        :return:
            A list of dictionaries faking the port_pairs
        """
    port_pairs = []
    for _ in range(count):
        port_pairs.append(FakeSfcPortPair.create_port_pair(attrs))
    return port_pairs