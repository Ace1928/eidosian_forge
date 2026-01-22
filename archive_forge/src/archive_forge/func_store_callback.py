from tqdm import tqdm, tqdm_notebook
from collections import OrderedDict
import time
def store_callback(self, **kw):
    """Execute something after the store has been updated by the given
        state elements.

        This default callback does nothing, overwrite it by subclassing
        """
    pass