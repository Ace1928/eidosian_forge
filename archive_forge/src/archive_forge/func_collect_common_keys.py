import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple
import wandb
from wandb.sdk.integration_utils.auto_logging import Response
from wandb.sdk.lib.runid import generate_id
def collect_common_keys(list_of_dicts: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    """Collect the common keys of a list of dictionaries. For each common key, put its values into a list in the order they appear in the original dictionaries.

    :param list_of_dicts: The list of dictionaries to inspect.
    :return: A dictionary with each common key and its corresponding list of values.
    """
    common_keys = set.intersection(*map(set, list_of_dicts))
    common_dict = {key: [] for key in common_keys}
    for d in list_of_dicts:
        for key in common_keys:
            common_dict[key].append(d[key])
    return common_dict