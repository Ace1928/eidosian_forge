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
def create_port_chain(attrs=None):
    """Create a fake port chain.

        :param Dictionary attrs:
            A dictionary with all attributes
        :return:
            A Dictionary with faking port chain attributes
        """
    attrs = attrs or {}
    port_chain_attrs = {'id': uuidutils.generate_uuid(), 'name': 'port-chain-name', 'description': 'description', 'port_pair_groups': uuidutils.generate_uuid(), 'flow_classifiers': uuidutils.generate_uuid(), 'chain_parameters': {'correlation': 'mpls', 'symmetric': False}, 'project_id': uuidutils.generate_uuid()}
    port_chain_attrs.update(attrs)
    return port_chain.SfcPortChain(**port_chain_attrs)