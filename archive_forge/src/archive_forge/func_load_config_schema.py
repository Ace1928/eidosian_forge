import json
import pathlib
def load_config_schema(key):
    """load a keyed filename"""
    return json.loads((CONFIGS / '{}.schema.json'.format(key)).read_text(encoding='utf-8'))