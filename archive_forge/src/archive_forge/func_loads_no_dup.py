import json
from typing import Any, Dict, Hashable, List, Tuple
def loads_no_dup(json_str: str) -> Any:
    """Load json string, and raise KeyError if there are duplicated keys

    :param json_str: json string
    :raises KeyError: if there are duplicated keys
    :return: the parsed object
    """
    return json.loads(json_str, object_pairs_hook=check_for_duplicate_keys)