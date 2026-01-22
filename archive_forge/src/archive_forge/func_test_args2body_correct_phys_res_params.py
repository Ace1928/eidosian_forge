import argparse
from datetime import datetime
from unittest import mock
from blazarclient import exception
from blazarclient import shell
from blazarclient import tests
from blazarclient.v1.shell_commands import leases
def test_args2body_correct_phys_res_params(self):
    args = argparse.Namespace(start='2020-07-24 20:00', end='2020-08-09 22:30', before_end='2020-08-09 21:30', events=[], name='lease-test', reservations=[], physical_reservations=['min=1,max=2,hypervisor_properties=["and", [">=", "$vcpus", "2"], [">=", "$memory_mb", "2048"]],resource_properties=["==", "$extra_key", "extra_value"],before_end=default'])
    expected = {'start': '2020-07-24 20:00', 'end': '2020-08-09 22:30', 'before_end': '2020-08-09 21:30', 'events': [], 'name': 'lease-test', 'reservations': [{'min': 1, 'max': 2, 'hypervisor_properties': '["and", [">=", "$vcpus", "2"], [">=", "$memory_mb", "2048"]]', 'resource_properties': '["==", "$extra_key", "extra_value"]', 'resource_type': 'physical:host', 'before_end': 'default'}]}
    self.assertDictEqual(self.cl.args2body(args), expected)