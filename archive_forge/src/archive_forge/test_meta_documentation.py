from unittest import mock
from openstack.cloud import meta
from openstack.compute.v2 import server as _server
from openstack import connection
from openstack.tests import fakes
from openstack.tests.unit import base
This test verifies that calling get_hostvars_froms_server
        ultimately calls list_server_security_groups, and that the return
        value from list_server_security_groups ends up in
        server['security_groups'].