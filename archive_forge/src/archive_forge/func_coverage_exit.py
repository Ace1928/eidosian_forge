from __future__ import (absolute_import, division, print_function)
def coverage_exit(*args, **kwargs):
    for instance in coverage_instances:
        instance.stop()
        instance.save()
    os_exit(*args, **kwargs)