import snappy
import sys
import spherogram.graphs as graphs
from spherogram import DTcodec
def test_Morwen():
    for dt in dtcodes:
        snappy.Manifold('DT[' + dt + ']')