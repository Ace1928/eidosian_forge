from unittest import mock
from openstackclient.compute.v2 import hypervisor_stats
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit import fakes
Create a fake hypervisor stats.

    :param dict attrs:
        A dictionary with all attributes
    :return:
        A dictionary that contains hypervisor stats information keys
    