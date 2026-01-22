import warnings
from ..helpers import quote_string, random_string, stringify_param_value
from .commands import AsyncGraphCommands, GraphCommands
from .edge import Edge  # noqa
from .node import Node  # noqa
from .path import Path  # noqa
def get_label(self, idx):
    """
        Returns a label by it's index

        Args:

        idx:
            The index of the label
        """
    try:
        label = self._labels[idx]
    except IndexError:
        self._refresh_labels()
        label = self._labels[idx]
    return label