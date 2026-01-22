import io
import os
import tempfile
import pytest
import networkx as nx
from networkx.readwrite.graphml import GraphMLWriter
from networkx.utils import edges_equal, nodes_equal
def test_graphml_header_line(self):
    good = '<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n<graphml xmlns="http://graphml.graphdrawing.org/xmlns"\n         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n         xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns\n         http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">\n  <key id="d0" for="node" attr.name="test" attr.type="boolean">\n    <default>false</default>\n  </key>\n  <graph id="G">\n    <node id="n0">\n      <data key="d0">true</data>\n    </node>\n  </graph>\n</graphml>\n'
    bad = '<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n<graphml>\n  <key id="d0" for="node" attr.name="test" attr.type="boolean">\n    <default>false</default>\n  </key>\n  <graph id="G">\n    <node id="n0">\n      <data key="d0">true</data>\n    </node>\n  </graph>\n</graphml>\n'
    ugly = '<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n<graphml xmlns="https://ghghgh">\n  <key id="d0" for="node" attr.name="test" attr.type="boolean">\n    <default>false</default>\n  </key>\n  <graph id="G">\n    <node id="n0">\n      <data key="d0">true</data>\n    </node>\n  </graph>\n</graphml>\n'
    for s in (good, bad):
        fh = io.BytesIO(s.encode('UTF-8'))
        G = nx.read_graphml(fh)
        H = nx.parse_graphml(s)
        for graph in [G, H]:
            assert graph.nodes['n0']['test']
    fh = io.BytesIO(ugly.encode('UTF-8'))
    pytest.raises(nx.NetworkXError, nx.read_graphml, fh)
    pytest.raises(nx.NetworkXError, nx.parse_graphml, ugly)