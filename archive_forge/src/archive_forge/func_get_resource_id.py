import sys
import time
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from oslo_serialization import jsonutils as json
from oslo_utils import timeutils
from oslo_utils import uuidutils
from saharaclient.api import base
def get_resource_id(manager, name_or_id):
    if uuidutils.is_uuid_like(name_or_id):
        return name_or_id
    else:
        return manager.find_unique(name=name_or_id).id