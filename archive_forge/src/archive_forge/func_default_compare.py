from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
from ansible.module_utils.common.dict_transformations import _snake_to_camel
from ansible.module_utils.six import string_types
def default_compare(self, modifiers, new, old, path, result):
    """
            Default dictionary comparison.
            This function will work well with most of the Azure resources.
            It correctly handles "location" comparison.

            Value handling:
                - if "new" value is None, it will be taken from "old" dictionary if "incremental_update"
                  is enabled.
            List handling:
                - if list contains "name" field it will be sorted by "name" before comparison is done.
                - if module has "incremental_update" set, items missing in the new list will be copied
                  from the old list

            Warnings:
                If field is marked as non-updatable, appropriate warning will be printed out and
                "new" structure will be updated to old value.

            :modifiers: Optional dictionary of modifiers, where key is the path and value is dict of modifiers
            :param new: New version
            :param old: Old version

            Returns True if no difference between structures has been detected.
            Returns False if difference was detected.
        """
    if new is None:
        return True
    elif isinstance(new, dict):
        comparison_result = True
        if not isinstance(old, dict):
            result['compare'].append('changed [' + path + '] old dict is null')
            comparison_result = False
        else:
            for k in set(new.keys()) | set(old.keys()):
                new_item = new.get(k, None)
                old_item = old.get(k, None)
                if new_item is None:
                    if isinstance(old_item, dict):
                        new[k] = old_item
                        result['compare'].append('new item was empty, using old [' + path + '][ ' + k + ' ]')
                elif not self.default_compare(modifiers, new_item, old_item, path + '/' + k, result):
                    comparison_result = False
        return comparison_result
    elif isinstance(new, list):
        comparison_result = True
        if not isinstance(old, list) or len(new) != len(old):
            result['compare'].append('changed [' + path + '] length is different or old value is null')
            comparison_result = False
        elif len(old) > 0:
            if isinstance(old[0], dict):
                key = None
                if 'id' in old[0] and 'id' in new[0]:
                    key = 'id'
                elif 'name' in old[0] and 'name' in new[0]:
                    key = 'name'
                else:
                    key = next(iter(old[0]))
                    new = sorted(new, key=lambda x: x.get(key, None))
                    old = sorted(old, key=lambda x: x.get(key, None))
            else:
                new = sorted(new)
                old = sorted(old)
            for i in range(len(new)):
                if not self.default_compare(modifiers, new[i], old[i], path + '/*', result):
                    comparison_result = False
        return comparison_result
    else:
        updatable = modifiers.get(path, {}).get('updatable', True)
        comparison = modifiers.get(path, {}).get('comparison', 'default')
        if comparison == 'ignore':
            return True
        elif comparison == 'default' or comparison == 'sensitive':
            if isinstance(old, string_types) and isinstance(new, string_types):
                new = new.lower()
                old = old.lower()
        elif comparison == 'location':
            if isinstance(old, string_types) and isinstance(new, string_types):
                new = new.replace(' ', '').lower()
                old = old.replace(' ', '').lower()
        if str(new) != str(old):
            result['compare'].append('changed [' + path + '] ' + str(new) + ' != ' + str(old) + ' - ' + str(comparison))
            if updatable:
                return False
            else:
                self.module.warn("property '" + path + "' cannot be updated (" + str(old) + '->' + str(new) + ')')
                return True
        else:
            return True