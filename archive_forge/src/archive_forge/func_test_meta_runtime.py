from __future__ import absolute_import, division, print_function
from awx.main.tests.functional.conftest import _request
from ansible.module_utils.six import string_types
import yaml
import os
import re
import glob
def test_meta_runtime():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
    meta_filename = 'meta/runtime.yml'
    module_dir = 'plugins/modules'
    print('\nMeta check:')
    with open('{0}/{1}'.format(base_dir, meta_filename), 'r') as f:
        meta_data_string = f.read()
    meta_data = yaml.load(meta_data_string, Loader=yaml.Loader)
    needs_grouping = []
    for file_name in glob.glob('{0}/{1}/*'.format(base_dir, module_dir)):
        if not os.path.isfile(file_name) or os.path.islink(file_name):
            continue
        with open(file_name, 'r') as f:
            if 'extends_documentation_fragment: awx.awx.auth' in f.read():
                needs_grouping.append(os.path.splitext(os.path.basename(file_name))[0])
    needs_to_be_removed = list(set(meta_data['action_groups']['controller']) - set(needs_grouping))
    needs_to_be_added = list(set(needs_grouping) - set(meta_data['action_groups']['controller']))
    needs_to_be_removed.sort()
    needs_to_be_added.sort()
    group = 'action-groups.controller'
    if needs_to_be_removed:
        print(cause_error('The following items should be removed from the {0} {1}:\n    {2}'.format(meta_filename, group, '\n    '.join(needs_to_be_removed))))
    if needs_to_be_added:
        print(cause_error('The following items should be added to the {0} {1}:\n    {2}'.format(meta_filename, group, '\n    '.join(needs_to_be_added))))