import datetime
import os
import re
from oslo_serialization import jsonutils as json
from blazarclient import exception
from blazarclient.i18n import _
def find_resource_id_by_name_or_id(client, resource_type, name_or_id, name_key, id_pattern):
    if re.match(id_pattern, name_or_id):
        return name_or_id
    return _find_resource_id_by_name(client, resource_type, name_or_id, name_key)