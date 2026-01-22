import pytest
import networkx as nx
from networkx.algorithms.assortativity.correlation import attribute_ac
from .base_test import BaseTestAttributeMixing, BaseTestDegreeMixing
Test degree assortativity with Pearson for a directed graph where
        the set of in/out degree does not equal the total degree.