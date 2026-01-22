import itertools
import os
import warnings
import pytest
import networkx as nx
def test_arrows():
    nx.draw_spring(barbell.to_directed())