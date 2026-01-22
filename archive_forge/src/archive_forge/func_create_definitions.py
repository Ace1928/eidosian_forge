import os
from typing import cast, Dict, Iterable, List, Optional, Union
from ansible.module_utils.six import string_types
from ansible.module_utils.urls import Request
def create_definitions(params: Dict) -> List[ResourceDefinition]:
    """Create a list of ResourceDefinitions from module inputs.

    This will take the module's inputs and return a list of ResourceDefintion
    objects. The resource definitions returned by this function should be as
    complete a definition as we can create based on the input. Any *List kinds
    will be removed and replaced by the resources contained in it.
    """
    if params.get('resource_definition'):
        d = cast(Union[str, List, Dict], params.get('resource_definition'))
        definitions = from_yaml(d)
    elif params.get('src'):
        d = cast(str, params.get('src'))
        if hasattr(d, 'startswith') and d.startswith(('https://', 'http://', 'ftp://')):
            data = Request().open('GET', d).read().decode('utf8')
            definitions = from_yaml(data)
        else:
            definitions = from_file(d)
    else:
        definitions = [{}]
    resource_definitions: List[Dict] = []
    for definition in definitions:
        merge_params(definition, params)
        kind = cast(Optional[str], definition.get('kind'))
        if kind and kind.endswith('List'):
            resource_definitions += flatten_list_kind(definition, params)
        else:
            resource_definitions.append(definition)
    return list(map(ResourceDefinition, resource_definitions))