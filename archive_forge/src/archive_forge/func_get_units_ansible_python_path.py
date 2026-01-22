from __future__ import annotations
import os
import sys
import typing as t
from ...constants import (
from ...io import (
from ...util import (
from ...util_common import (
from ...ansible_util import (
from ...target import (
from ...config import (
from ...coverage_util import (
from ...data import (
from ...executor import (
from ...python_requirements import (
from ...content_config import (
from ...host_configs import (
from ...provisioning import (
from ...pypi_proxy import (
from ...host_profiles import (
def get_units_ansible_python_path(args: UnitsConfig, test_context: str) -> str:
    """
    Return a directory usable for PYTHONPATH, containing only the modules and module_utils portion of the ansible package.
    The temporary directory created will be cached for the lifetime of the process and cleaned up at exit.
    """
    if test_context == TestContext.controller:
        return get_ansible_python_path(args)
    try:
        cache = get_units_ansible_python_path.cache
    except AttributeError:
        cache = get_units_ansible_python_path.cache = {}
    python_path = cache.get(test_context)
    if python_path:
        return python_path
    python_path = create_temp_dir(prefix='ansible-test-')
    ansible_path = os.path.join(python_path, 'ansible')
    ansible_test_path = os.path.join(python_path, 'ansible_test')
    write_text_file(os.path.join(ansible_path, '__init__.py'), '', True)
    os.symlink(os.path.join(ANSIBLE_LIB_ROOT, 'module_utils'), os.path.join(ansible_path, 'module_utils'))
    if data_context().content.collection:
        make_dirs(os.path.join(ansible_path, 'config'))
        os.symlink(os.path.join(ANSIBLE_LIB_ROOT, 'config', 'ansible_builtin_runtime.yml'), os.path.join(ansible_path, 'config', 'ansible_builtin_runtime.yml'))
        write_text_file(os.path.join(ansible_path, 'utils', '__init__.py'), '', True)
        os.symlink(os.path.join(ANSIBLE_LIB_ROOT, 'utils', 'collection_loader'), os.path.join(ansible_path, 'utils', 'collection_loader'))
        write_text_file(os.path.join(ansible_test_path, '__init__.py'), '', True)
        write_text_file(os.path.join(ansible_test_path, '_internal', '__init__.py'), '', True)
    elif test_context == TestContext.modules:
        os.symlink(os.path.join(ANSIBLE_LIB_ROOT, 'modules'), os.path.join(ansible_path, 'modules'))
    cache[test_context] = python_path
    return python_path