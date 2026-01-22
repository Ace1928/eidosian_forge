from dash.exceptions import InvalidCallbackReturnValue
from ._utils import AttributeDict, stringify_id
def update_args_group(g, triggered):
    if isinstance(g, dict):
        str_id = stringify_id(g['id'])
        prop_id = f'{str_id}.{g['property']}'
        new_values = {'value': g.get('value'), 'str_id': str_id, 'triggered': prop_id in triggered, 'id': AttributeDict(g['id']) if isinstance(g['id'], dict) else g['id']}
        g.update(new_values)