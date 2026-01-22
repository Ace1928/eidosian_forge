import bz2
import importlib.resources
import os
import pickle
import pytest
import networkx as nx
Negative selfloops should cause an exception if uncapacitated and
        always be saturated otherwise.
        