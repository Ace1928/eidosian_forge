from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.hpe.nimble.plugins.module_utils.hpe_nimble import __version__ as NIMBLE_ANSIBLE_VERSION
import ansible_collections.hpe.nimble.plugins.module_utils.hpe_nimble as utils
def update_shelve(client_obj, shelf_serial, **kwargs):
    if utils.is_null_or_empty(shelf_serial):
        return (False, False, 'Shelf update failed as no shelf id provided.', {})
    try:
        shelf_list_resp = client_obj.shelves.list(detail=True)
        if utils.is_null_or_empty(shelf_list_resp):
            return (False, False, f"Shelf serial '{shelf_serial}' is not present on array.", {})
        else:
            shelf_resp = None
            for resp in shelf_list_resp:
                if shelf_serial == resp.attrs.get('serial'):
                    shelf_resp = resp
                    break
            if utils.is_null_or_empty(shelf_resp):
                return (False, False, f"Shelf serial '{shelf_serial}' is not present on array.", {})
            else:
                changed_attrs_dict, params = utils.remove_unchanged_or_null_args(shelf_resp, **kwargs)
                if changed_attrs_dict.__len__() > 0:
                    shelf_resp = client_obj.shelves.update(id=shelf_resp.attrs.get('id'), **params)
                    return (True, True, f"Successfully updated Shelf '{shelf_serial}'.", shelf_resp.attrs)
                else:
                    return (True, False, f"Shelf serial '{shelf_serial}' already updated.", shelf_resp.attrs)
    except Exception as e:
        return (False, False, 'Shelf update failed | %s' % str(e), {})