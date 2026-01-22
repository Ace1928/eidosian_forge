import subprocess
import warnings
from collections import defaultdict
from itertools import chain
from pprint import pformat
from nltk.internals import find_binary
from nltk.tree import Tree
def remove_by_address(self, address):
    """
        Removes the node with the given address.  References
        to this node in others will still exist.
        """
    del self.nodes[address]