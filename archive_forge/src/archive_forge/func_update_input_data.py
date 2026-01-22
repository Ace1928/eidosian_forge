import ast
import tenacity
from oslo_log import log as logging
from heat.common import exception
from heat.objects import sync_point as sync_point_object
def update_input_data(context, entity_id, current_traversal, is_update, atomic_key, input_data):
    rows_updated = sync_point_object.SyncPoint.update_input_data(context, entity_id, current_traversal, is_update, atomic_key, input_data)
    return rows_updated