import argparse
import base64
import builtins
import collections
import datetime
import io
import os
from unittest import mock
import fixtures
from oslo_utils import timeutils
import testtools
import novaclient
from novaclient import api_versions
from novaclient import base
import novaclient.client
from novaclient import exceptions
import novaclient.shell
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
from novaclient.v2 import servers
import novaclient.v2.shell
def test_boot_no_image_bdms_v2(self):
    self.run_command('boot --flavor 1 --block-device id=fake-id,source=volume,dest=volume,bus=virtio,device=vda,size=1,format=ext4,bootindex=0,type=disk,shutdown=preserve some-server')
    self.assert_called_anytime('POST', '/servers', {'server': {'flavorRef': '1', 'name': 'some-server', 'block_device_mapping_v2': [{'uuid': 'fake-id', 'source_type': 'volume', 'destination_type': 'volume', 'disk_bus': 'virtio', 'device_name': 'vda', 'volume_size': '1', 'guest_format': 'ext4', 'boot_index': '0', 'device_type': 'disk', 'delete_on_termination': False}], 'imageRef': '', 'min_count': 1, 'max_count': 1}})
    cmd = 'boot --flavor 1 --boot-volume fake-id some-server'
    self.run_command(cmd)
    self.assert_called_anytime('POST', '/servers', {'server': {'flavorRef': '1', 'name': 'some-server', 'block_device_mapping_v2': [{'uuid': 'fake-id', 'source_type': 'volume', 'destination_type': 'volume', 'boot_index': 0, 'delete_on_termination': False}], 'imageRef': '', 'min_count': 1, 'max_count': 1}})
    cmd = 'boot --flavor 1 --snapshot fake-id some-server'
    self.run_command(cmd)
    self.assert_called_anytime('POST', '/servers', {'server': {'flavorRef': '1', 'name': 'some-server', 'block_device_mapping_v2': [{'uuid': 'fake-id', 'source_type': 'snapshot', 'destination_type': 'volume', 'boot_index': 0, 'delete_on_termination': False}], 'imageRef': '', 'min_count': 1, 'max_count': 1}})
    self.run_command('boot --flavor 1 --swap 1 some-server')
    self.assert_called_anytime('POST', '/servers', {'server': {'flavorRef': '1', 'name': 'some-server', 'block_device_mapping_v2': [{'source_type': 'blank', 'destination_type': 'local', 'boot_index': -1, 'guest_format': 'swap', 'volume_size': '1', 'delete_on_termination': True}], 'imageRef': '', 'min_count': 1, 'max_count': 1}})
    self.run_command('boot --flavor 1 --ephemeral size=1,format=ext4 some-server')
    self.assert_called_anytime('POST', '/servers', {'server': {'flavorRef': '1', 'name': 'some-server', 'block_device_mapping_v2': [{'source_type': 'blank', 'destination_type': 'local', 'boot_index': -1, 'guest_format': 'ext4', 'volume_size': '1', 'delete_on_termination': True}], 'imageRef': '', 'min_count': 1, 'max_count': 1}})