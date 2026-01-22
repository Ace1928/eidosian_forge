import copy
import functools
import uuid
from oslo_config import cfg
from oslo_log import log as logging
from glance.common import exception
from glance.common import timeutils
from glance.common import utils
from glance.db import utils as db_utils
from glance.i18n import _, _LI, _LW
@log_call
def image_update(context, image_id, image_values, purge_props=False, from_state=None, v1_mode=False, atomic_props=None):
    global DATA
    try:
        image = DATA['images'][image_id]
    except KeyError:
        raise exception.ImageNotFound(image_id)
    location_data = image_values.pop('locations', None)
    if location_data is not None:
        _image_locations_set(context, image_id, location_data)
    if atomic_props is None:
        atomic_props = []
    new_properties = image_values.pop('properties', {})
    for prop in image['properties']:
        if prop['name'] in atomic_props:
            continue
        elif prop['name'] in new_properties:
            prop['value'] = new_properties.pop(prop['name'])
        elif purge_props:
            prop['deleted'] = True
    image['updated_at'] = timeutils.utcnow()
    _image_update(image, image_values, {k: v for k, v in new_properties.items() if k not in atomic_props})
    DATA['images'][image_id] = image
    image = _normalize_locations(context, copy.deepcopy(image))
    if v1_mode:
        image = db_utils.mutate_image_dict_to_v1(image)
    return image