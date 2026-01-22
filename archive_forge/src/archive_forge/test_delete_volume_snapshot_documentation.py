from openstack.cloud import meta
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base

        Test that a timeout while waiting for the volume snapshot to delete
        raises an exception in delete_volume_snapshot.
        