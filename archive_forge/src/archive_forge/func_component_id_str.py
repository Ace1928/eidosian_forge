import json
from dash.development.base_component import Component
from ._validate import validate_callback
from ._grouping import flatten_grouping, make_grouping_by_index
def component_id_str(self):
    i = self.component_id

    def _dump(v):
        return json.dumps(v, sort_keys=True, separators=(',', ':'))

    def _json(k, v):
        vstr = v.to_json() if hasattr(v, 'to_json') else json.dumps(v)
        return f'{json.dumps(k)}:{vstr}'
    if isinstance(i, dict):
        return '{' + ','.join((_json(k, i[k]) for k in sorted(i))) + '}'
    return i