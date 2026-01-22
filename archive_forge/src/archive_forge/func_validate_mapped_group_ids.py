import ast
import copy
import re
import flask
import jsonschema
from oslo_config import cfg
from oslo_log import log
from oslo_serialization import jsonutils
from oslo_utils import timeutils
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.i18n import _
def validate_mapped_group_ids(group_ids, mapping_id, identity_api):
    """Iterate over group ids and make sure they are present in the backend.

    This call is not transactional.
    :param group_ids: IDs of the groups to be checked
    :type group_ids: list of str

    :param mapping_id: id of the mapping used for this operation
    :type mapping_id: str

    :param identity_api: Identity Manager object used for communication with
                         backend
    :type identity_api: identity.Manager

    :raises keystone.exception.MappedGroupNotFound: If the group returned by
        mapping was not found in the backend.

    """
    for group_id in group_ids:
        try:
            identity_api.get_group(group_id)
        except exception.GroupNotFound:
            raise exception.MappedGroupNotFound(group_id=group_id, mapping_id=mapping_id)