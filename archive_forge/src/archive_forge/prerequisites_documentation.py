from __future__ import absolute_import, division, print_function
import re
import importlib
import importlib.metadata

    Tries to import a module and optionally validates its package version.
    Calls AnsibleModule.fail_json() if not satisfied.

    :param ansible_module: an AnsibleModule instance
    :param module: a string with module name to try to import
    :param package: a string, package to check version for; must be specified with 'version_requirements'
    :param version_requirements: a string, version requirements for 'package'
    