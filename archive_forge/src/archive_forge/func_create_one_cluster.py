import argparse
import copy
import datetime
import uuid
from magnumclient.tests.osc.unit import osc_fakes
from magnumclient.tests.osc.unit import osc_utils
@staticmethod
def create_one_cluster(attrs=None):
    """Create a fake Cluster.

        :param Dictionary attrs:
            A dictionary with all attributes
        :return:
            A FakeResource object, with flavor_id, image_id, and so on
        """
    attrs = attrs or {}
    cluster_info = {'status': 'CREATE_IN_PROGRESS', 'health_status': 'HEALTHY', 'cluster_template_id': 'fake-ct', 'node_addresses': [], 'uuid': '3a369884-b6ba-484f-a206-919b4b718aff', 'stack_id': 'c4554582-77bd-4734-8f1a-72c3c40e5fb4', 'status_reason': None, 'labels': {}, 'labels_overridden': {}, 'labels_added': {}, 'labels_skipped': {}, 'fixed_network': 'fixed-network', 'fixed_subnet': 'fixed-subnet', 'floating_ip_enabled': True, 'created_at': '2017-03-16T18:40:39+00:00', 'updated_at': '2017-03-16T18:40:45+00:00', 'coe_version': None, 'faults': None, 'keypair': 'fakekey', 'api_address': None, 'master_addresses': [], 'create_timeout': 60, 'node_count': 1, 'discovery_url': 'https://fake.cluster', 'docker_volume_size': 1, 'master_count': 1, 'container_version': None, 'name': 'fake-cluster', 'master_flavor_id': None, 'flavor_id': 'm1.medium', 'project_id': None, 'health_status_reason': {'api': 'ok'}, 'master_lb_enabled': False}
    cluster_info.update(attrs)
    cluster = osc_fakes.FakeResource(info=copy.deepcopy(cluster_info), loaded=True)
    return cluster