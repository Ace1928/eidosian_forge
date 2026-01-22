def load_custom_schema_data():
    import os.path
    import json
    json_file = os.path.join(os.path.dirname(__file__), 'debugProtocolCustom.json')
    with open(json_file, 'rb') as json_contents:
        json_schema_data = json.loads(json_contents.read())
    return json_schema_data