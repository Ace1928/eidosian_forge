from unittest import mock
from novaclient import client as base_client
from novaclient import exceptions as nova_exceptions
import requests
from urllib import parse as urlparse
from heat.tests import fakes
def get_images_detail(self, **kw):
    return (200, {'images': [{'id': 1, 'name': 'CentOS 5.2', 'updated': '2010-10-10T12:00:00Z', 'created': '2010-08-10T12:00:00Z', 'status': 'ACTIVE', 'metadata': {'test_key': 'test_value'}, 'links': {}}, {'id': 743, 'name': 'My Server Backup', 'serverId': 1234, 'updated': '2010-10-10T12:00:00Z', 'created': '2010-08-10T12:00:00Z', 'status': 'SAVING', 'progress': 80, 'links': {}}, {'id': 744, 'name': 'F17-x86_64-gold', 'serverId': 9999, 'updated': '2010-10-10T12:00:00Z', 'created': '2010-08-10T12:00:00Z', 'status': 'SAVING', 'progress': 80, 'links': {}}, {'id': 745, 'name': 'F17-x86_64-cfntools', 'serverId': 9998, 'updated': '2010-10-10T12:00:00Z', 'created': '2010-08-10T12:00:00Z', 'status': 'SAVING', 'progress': 80, 'links': {}}, {'id': 746, 'name': 'F20-x86_64-cfntools', 'serverId': 9998, 'updated': '2010-10-10T12:00:00Z', 'created': '2010-08-10T12:00:00Z', 'status': 'SAVING', 'progress': 80, 'links': {}}]})