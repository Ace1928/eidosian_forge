import json
import os
import numpy as np
def get_key_from_id(id: str) -> str:
    """
    Gets the key from an id.
    :param id:
    :return:
    """
    assert id in KEYMAP, 'ID not found'
    return KEYMAP[id]