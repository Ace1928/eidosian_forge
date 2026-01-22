from __future__ import absolute_import, division, print_function
from copy import deepcopy
from ansible.module_utils._text import to_bytes
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.six import string_types
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.junipernetworks.junos.plugins.module_utils.network.junos.argspec.logging_global.logging_global import (
def parse_file_node(self, conf):
    files_list = []
    files = []
    if isinstance(conf, dict):
        files.append(conf)
    else:
        files = conf
    for file in files:
        file_dict = {}
        file_dict['name'] = file.get('name')
        if 'allow-duplicates' in file:
            file_dict['allow_duplicates'] = True
        if 'contents' in file.keys():
            contents = file.get('contents')
            if isinstance(contents, list):
                for content in contents:
                    file_dict = self.parse_console_node(content, file_dict)
            else:
                file_dict = self.parse_console_node(contents, file_dict)
        if 'archive' in file.keys():
            archive_dict = self.parse_archive_node(file.get('archive'))
            file_dict['archive'] = archive_dict
        if 'explicit-priority' in file.keys():
            file_dict['explicit_priority'] = True
        if 'match' in file.keys():
            file_dict['match'] = file.get('match')
        if 'match-strings' in file.keys():
            match_strings = file.get('match-strings')
            match_strings_list = []
            if isinstance(match_strings, list):
                for item in match_strings:
                    match_strings_list.append(item)
            else:
                match_strings_list.append(match_strings)
            file_dict['match_strings'] = match_strings_list
        if 'structured-data' in file.keys():
            structured_data_dict = {}
            if file.get('structured-data'):
                structured_data_dict['brief'] = True
            else:
                structured_data_dict['set'] = True
            file_dict['structured_data'] = structured_data_dict
        files_list.append(file_dict)
    return files_list