import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple
import wandb
from wandb.sdk.integration_utils.auto_logging import Response
from wandb.sdk.lib.runid import generate_id
def reorder_and_convert_dict_list_to_table(data: List[Dict[str, Any]], order: List[str]) -> Tuple[List[str], List[List[Any]]]:
    """Convert a list of dictionaries to a pair of column names and corresponding values, with the option to order specific dictionaries.

    :param data: A list of dictionaries.
    :param order: A list of keys specifying the desired order for specific dictionaries. The remaining dictionaries will be ordered based on their original order.
    :return: A pair of column names and corresponding values.
    """
    final_columns = []
    keys_present = set()
    for key in order:
        if key not in keys_present:
            final_columns.append(key)
            keys_present.add(key)
    for d in data:
        for key in d:
            if key not in keys_present:
                final_columns.append(key)
                keys_present.add(key)
    values = []
    for d in data:
        row = []
        for key in final_columns:
            row.append(d.get(key, None))
        values.append(row)
    return (final_columns, values)