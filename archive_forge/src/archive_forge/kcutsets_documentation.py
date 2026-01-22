import copy
from collections import defaultdict
from itertools import combinations
from operator import itemgetter
import networkx as nx
from networkx.algorithms.flow import (
from .utils import build_auxiliary_node_connectivity
Assumes that the input graph is connected