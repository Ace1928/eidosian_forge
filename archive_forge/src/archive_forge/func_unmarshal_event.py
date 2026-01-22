import json
import pydoc
from kubernetes import client
def unmarshal_event(self, data, return_type):
    try:
        js = json.loads(data)
    except ValueError:
        return data
    js['raw_object'] = js['object']
    if return_type:
        obj = SimpleNamespace(data=json.dumps(js['raw_object']))
        js['object'] = self._api_client.deserialize(obj, return_type)
        if hasattr(js['object'], 'metadata'):
            self.resource_version = js['object'].metadata.resource_version
        elif isinstance(js['object'], dict) and 'metadata' in js['object'] and ('resourceVersion' in js['object']['metadata']):
            self.resource_version = js['object']['metadata']['resourceVersion']
    return js