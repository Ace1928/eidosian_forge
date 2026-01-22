import heapq
import inspect
import unittest
from pbr.version import VersionInfo
def cost_of_switching(self, old_resource_set, new_resource_set):
    """Cost of switching from 'old_resource_set' to 'new_resource_set'.

        This is calculated by adding the cost of tearing down unnecessary
        resources to the cost of setting up the newly-needed resources.

        Note that resources which are always dirtied may skew the predicted
        skew the cost of switching because they are considered common, even
        when reusing them may actually be equivalent to a teardown+setup
        operation.
        """
    new_resources = new_resource_set - old_resource_set
    gone_resources = old_resource_set - new_resource_set
    return sum((resource.setUpCost for resource in new_resources)) + sum((resource.tearDownCost for resource in gone_resources))