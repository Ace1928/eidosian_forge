import argparse
import copy
import datetime
import uuid
from magnumclient.tests.osc.unit import osc_fakes
from magnumclient.tests.osc.unit import osc_utils
@staticmethod
def create_cluster_templates(attrs=None, count=2):
    """Create multiple fake cluster templates.

        :param Dictionary attrs:
            A dictionary with all attributes
        :param int count:
            The number of cluster templates to fake
        :return:
            A list of FakeResource objects faking the cluster templates
        """
    cts = []
    for i in range(0, count):
        cts.append(FakeClusterTemplate.create_one_cluster_template(attrs))
    return cts