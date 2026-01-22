from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
def ovirtdiff(self, vm1, vm2):
    """
        This filter takes two dictionaries of two different resources and compare
        them. It return dictionari with keys 'before' and 'after', where 'before'
        containes old values of resources and 'after' contains new values.
        This is mainly good to compare current VM object and next run VM object to see
        the difference for the next_run.
        """
    before = []
    after = []
    if vm1.get('next_run_configuration_exists'):
        keys = [key for key in set(list(vm1.keys()) + list(vm2.keys())) if key in vm1 and (key not in vm2 or vm2[key] != vm1[key]) or (key in vm2 and (key not in vm1 or vm1[key] != vm2[key]))]
        for key in keys:
            before.append((key, vm1.get(key)))
            after.append((key, vm2.get(key, vm1.get(key))))
    return {'before': dict(before), 'after': dict(after)}