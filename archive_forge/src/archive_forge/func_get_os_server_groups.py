import copy
import datetime
import re
from unittest import mock
from urllib import parse
from oslo_utils import strutils
import novaclient
from novaclient import api_versions
from novaclient import client as base_client
from novaclient import exceptions
from novaclient.tests.unit import fakes
from novaclient.tests.unit import utils
from novaclient.v2 import client
def get_os_server_groups(self, **kw):
    server_groups = [{'members': [], 'metadata': {}, 'id': '2cbd51f4-fafe-4cdb-801b-cf913a6f288b', 'policies': [], 'name': 'ig1'}, {'members': [], 'metadata': {}, 'id': '4473bb03-4370-4bfb-80d3-dc8cffc47d94', 'policies': ['anti-affinity'], 'name': 'ig2'}, {'members': [], 'metadata': {'key': 'value'}, 'id': '31ab9bdb-55e1-4ac3-b094-97eeb1b65cc4', 'policies': [], 'name': 'ig3'}, {'members': ['2dccb4a1-02b9-482a-aa23-5799490d6f5d'], 'metadata': {}, 'id': '4890bb03-7070-45fb-8453-d34556c87d94', 'policies': ['anti-affinity'], 'name': 'ig2'}]
    other_project_server_groups = [{'members': [], 'metadata': {}, 'id': '11111111-1111-1111-1111-111111111111', 'policies': [], 'name': 'ig4'}, {'members': [], 'metadata': {}, 'id': '22222222-2222-2222-2222-222222222222', 'policies': ['anti-affinity'], 'name': 'ig5'}, {'members': [], 'metadata': {'key': 'value'}, 'id': '31ab9bdb-55e1-4ac3-b094-97eeb1b65cc4', 'policies': [], 'name': 'ig6'}, {'members': ['33333333-3333-3333-3333-333333333333'], 'metadata': {}, 'id': '44444444-4444-4444-4444-444444444444', 'policies': ['anti-affinity'], 'name': 'ig5'}]
    if kw.get('all_projects', False):
        server_groups.extend(other_project_server_groups)
    limit = int(kw.get('limit', 1000))
    offset = int(kw.get('offset', 0))
    server_groups = server_groups[offset:limit + 1]
    return (200, {}, {'server_groups': server_groups})