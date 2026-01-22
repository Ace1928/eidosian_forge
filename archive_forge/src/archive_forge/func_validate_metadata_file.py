from __future__ import annotations
import datetime
import os
import re
import sys
from functools import partial
import yaml
from voluptuous import All, Any, MultipleInvalid, PREVENT_EXTRA
from voluptuous import Required, Schema, Invalid
from voluptuous.humanize import humanize_error
from ansible.module_utils.compat.version import StrictVersion, LooseVersion
from ansible.module_utils.six import string_types
from ansible.utils.collection_loader import AnsibleCollectionRef
from ansible.utils.version import SemanticVersion
def validate_metadata_file(path, is_ansible, check_deprecation_dates=False):
    """Validate explicit runtime metadata file"""
    try:
        with open(path, 'r', encoding='utf-8') as f_path:
            routing = yaml.safe_load(f_path)
    except yaml.error.MarkedYAMLError as ex:
        print('%s:%d:%d: YAML load failed: %s' % (path, ex.context_mark.line + 1 if ex.context_mark else 0, ex.context_mark.column + 1 if ex.context_mark else 0, re.sub('\\s+', ' ', str(ex))))
        return
    except Exception as ex:
        print('%s:%d:%d: YAML load failed: %s' % (path, 0, 0, re.sub('\\s+', ' ', str(ex))))
        return
    if is_ansible:
        current_version = get_ansible_version()
    else:
        current_version = get_collection_version()
    avoid_additional_data = Schema(Any({Required('removal_version'): any_value, 'warning_text': any_value}, {Required('removal_date'): any_value, 'warning_text': any_value}), extra=PREVENT_EXTRA)
    deprecation_schema = All(Schema({'removal_version': partial(removal_version, is_ansible=is_ansible, current_version=current_version), 'removal_date': partial(isodate, check_deprecation_date=check_deprecation_dates), 'warning_text': Any(*string_types)}), avoid_additional_data)
    tombstoning_schema = All(Schema({'removal_version': partial(removal_version, is_ansible=is_ansible, current_version=current_version, is_tombstone=True), 'removal_date': partial(isodate, is_tombstone=True), 'warning_text': Any(*string_types)}), avoid_additional_data)
    plugins_routing_common_schema = Schema({'deprecation': Any(deprecation_schema), 'tombstone': Any(tombstoning_schema), 'redirect': fqcr}, extra=PREVENT_EXTRA)
    plugin_routing_schema = Any(plugins_routing_common_schema)
    plugin_routing_schema_modules = Any(plugins_routing_common_schema.extend({'action_plugin': fqcr}))
    plugin_routing_schema_mu = Any(plugins_routing_common_schema.extend({'redirect': Any(*string_types)}))
    list_dict_plugin_routing_schema = [{str_type: plugin_routing_schema} for str_type in string_types]
    list_dict_plugin_routing_schema_mu = [{str_type: plugin_routing_schema_mu} for str_type in string_types]
    list_dict_plugin_routing_schema_modules = [{str_type: plugin_routing_schema_modules} for str_type in string_types]
    plugin_schema = Schema({'action': Any(None, *list_dict_plugin_routing_schema), 'become': Any(None, *list_dict_plugin_routing_schema), 'cache': Any(None, *list_dict_plugin_routing_schema), 'callback': Any(None, *list_dict_plugin_routing_schema), 'cliconf': Any(None, *list_dict_plugin_routing_schema), 'connection': Any(None, *list_dict_plugin_routing_schema), 'doc_fragments': Any(None, *list_dict_plugin_routing_schema), 'filter': Any(None, *list_dict_plugin_routing_schema), 'httpapi': Any(None, *list_dict_plugin_routing_schema), 'inventory': Any(None, *list_dict_plugin_routing_schema), 'lookup': Any(None, *list_dict_plugin_routing_schema), 'module_utils': Any(None, *list_dict_plugin_routing_schema_mu), 'modules': Any(None, *list_dict_plugin_routing_schema_modules), 'netconf': Any(None, *list_dict_plugin_routing_schema), 'shell': Any(None, *list_dict_plugin_routing_schema), 'strategy': Any(None, *list_dict_plugin_routing_schema), 'terminal': Any(None, *list_dict_plugin_routing_schema), 'test': Any(None, *list_dict_plugin_routing_schema), 'vars': Any(None, *list_dict_plugin_routing_schema)}, extra=PREVENT_EXTRA)
    import_redirection_schema = Any(Schema({'redirect': Any(*string_types)}, extra=PREVENT_EXTRA))
    list_dict_import_redirection_schema = [{str_type: import_redirection_schema} for str_type in string_types]
    schema = Schema({'plugin_routing': Any(plugin_schema), 'import_redirection': Any(None, *list_dict_import_redirection_schema), 'requires_ansible': Any(*string_types), 'action_groups': dict}, extra=PREVENT_EXTRA)
    try:
        schema(routing)
    except MultipleInvalid as ex:
        for error in ex.errors:
            print('%s:%d:%d: %s' % (path, 0, 0, humanize_error(routing, error)))