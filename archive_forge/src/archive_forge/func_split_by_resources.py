import heapq
import inspect
import unittest
from pbr.version import VersionInfo
def split_by_resources(tests):
    """Split a list of tests by the resources that the tests use.

    :return: a dictionary mapping sets of resources to lists of tests
    using that combination of resources.  The dictionary always
    contains an entry for "no resources".
    """
    no_resources = frozenset()
    resource_set_tests = {no_resources: []}
    for test in tests:
        resources = getattr(test, 'resources', ())
        all_resources = list((resource.neededResources() for _, resource in resources))
        resource_set = set()
        for resource_list in all_resources:
            resource_set.update(resource_list)
        resource_set_tests.setdefault(frozenset(resource_set), []).append(test)
    return resource_set_tests