from . import base
from cliff import columns
def emit_list(self, column_names, data, stdout, parsed_args):
    import yaml
    items = []
    for item in data:
        items.append({n: _yaml_friendly(i) for n, i in zip(column_names, item)})
    yaml.safe_dump(items, stream=stdout, default_flow_style=False)