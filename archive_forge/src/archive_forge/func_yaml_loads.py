from typing import Union, IO, Any
from io import StringIO
import sys
from .ruamel_yaml import YAML
from .ruamel_yaml.representer import RepresenterError
from .util import force_path, FilePath, YAMLInput, YAMLOutput
def yaml_loads(data: Union[str, IO]) -> YAMLOutput:
    """Deserialize unicode or a file object a Python object.

    data (str / file): The data to deserialize.
    RETURNS: The deserialized Python object.
    """
    yaml = CustomYaml()
    try:
        return yaml.load(data)
    except Exception as e:
        raise ValueError(f'Invalid YAML: {e}')