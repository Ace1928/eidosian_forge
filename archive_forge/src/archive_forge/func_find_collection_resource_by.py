from __future__ import (absolute_import, division, print_function)
import os
import traceback
from ansible.module_utils.basic import missing_required_lib
def find_collection_resource_by(self, collection_name, **params):
    """ Searches the collection resource by the collection name and the param passed.

        Returns:
            the resource as an object if it exists in manageiq, None otherwise.
        """
    try:
        entity = self.client.collections.__getattribute__(collection_name).get(**params)
    except ValueError:
        return None
    except Exception as e:
        self.module.fail_json(msg='failed to find resource {error}'.format(error=e))
    return vars(entity)