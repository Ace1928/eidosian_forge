from __future__ import (absolute_import, division, print_function)
import importlib
import os
import re
import sys
import textwrap
import yaml
def get_provider_informations(providers):
    files_to_remove = []

    def add_init_py(path):
        path = os.path.join(path, '__init__.py')
        if os.path.exists(path):
            return
        with open(path, 'wb') as f:
            f.write(b'')
        files_to_remove.append(path)
    try:
        sys.path.append(os.path.join('..', '..', '..'))
        add_init_py(os.path.join('..', '..'))
        add_init_py(os.path.join('..'))
        add_init_py(os.path.join('.'))
        add_init_py(os.path.join('plugins'))
        add_init_py(os.path.join('plugins', 'module_utils'))
        provider_infos = {}
        errors = []
        for provider in providers:
            add_init_py(os.path.join('plugins', 'module_utils', provider))
            full_py_path = 'ansible_collections.community.dns.plugins.module_utils.{0}.api'.format(provider)
            full_pathname = os.path.join('plugins', 'module_utils', provider, 'api.py')
            try:
                loader = importlib.machinery.SourceFileLoader(full_py_path, full_pathname)
                spec = importlib.util.spec_from_loader(full_py_path, loader)
                the_module = importlib.util.module_from_spec(spec)
                loader.exec_module(the_module)
            except Exception as e:
                errors.append('{0}: Error while importing module {1}: {2}'.format(full_pathname, full_py_path, e))
                continue
            create_provider_info_fn_name = 'create_{0}_provider_information'.format(provider)
            try:
                create_provider_info_fn = provider_information = the_module.__dict__[create_provider_info_fn_name]
                provider_infos[provider] = create_provider_info_fn()
            except KeyError as e:
                errors.append('{0}: Cannot find function {1}'.format(full_pathname, create_provider_info_fn))
            except Exception as e:
                errors.append('{0}: Error while invoking function {1}: {2}'.format(full_pathname, create_provider_info_fn, e))
        return (provider_infos, errors)
    finally:
        for path in files_to_remove:
            os.remove(path)