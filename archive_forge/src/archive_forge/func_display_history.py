import copy
import json
import pickle
import traceback
import parlai.utils.logging as logging
from typing import List
def display_history(self, key):
    """
        Display the history for an item in the dict.
        """
    changes = []
    i = 0
    for key_, val, loc in self.history:
        if key != key_:
            continue
        i += 1
        changes.append(f'{i}. {key} was set to {val} at:\n{loc}')
    if changes:
        return '\n'.join(changes)
    else:
        return f'No history for {key}'