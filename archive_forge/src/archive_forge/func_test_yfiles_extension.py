import io
import os
import tempfile
import pytest
import networkx as nx
from networkx.readwrite.graphml import GraphMLWriter
from networkx.utils import edges_equal, nodes_equal
def test_yfiles_extension(self):
    data = '<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n<graphml xmlns="http://graphml.graphdrawing.org/xmlns"\n         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n         xmlns:y="http://www.yworks.com/xml/graphml"\n         xmlns:yed="http://www.yworks.com/xml/yed/3"\n         xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns\n         http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">\n  <!--Created by yFiles for Java 2.7-->\n  <key for="graphml" id="d0" yfiles.type="resources"/>\n  <key attr.name="url" attr.type="string" for="node" id="d1"/>\n  <key attr.name="description" attr.type="string" for="node" id="d2"/>\n  <key for="node" id="d3" yfiles.type="nodegraphics"/>\n  <key attr.name="Description" attr.type="string" for="graph" id="d4">\n    <default/>\n  </key>\n  <key attr.name="url" attr.type="string" for="edge" id="d5"/>\n  <key attr.name="description" attr.type="string" for="edge" id="d6"/>\n  <key for="edge" id="d7" yfiles.type="edgegraphics"/>\n  <graph edgedefault="directed" id="G">\n    <node id="n0">\n      <data key="d3">\n        <y:ShapeNode>\n          <y:Geometry height="30.0" width="30.0" x="125.0" y="100.0"/>\n          <y:Fill color="#FFCC00" transparent="false"/>\n          <y:BorderStyle color="#000000" type="line" width="1.0"/>\n          <y:NodeLabel alignment="center" autoSizePolicy="content"\n           borderDistance="0.0" fontFamily="Dialog" fontSize="13"\n           fontStyle="plain" hasBackgroundColor="false" hasLineColor="false"\n           height="19.1328125" modelName="internal" modelPosition="c"\n           textColor="#000000" visible="true" width="12.27099609375"\n           x="8.864501953125" y="5.43359375">1</y:NodeLabel>\n          <y:Shape type="rectangle"/>\n        </y:ShapeNode>\n      </data>\n    </node>\n    <node id="n1">\n      <data key="d3">\n        <y:ShapeNode>\n          <y:Geometry height="30.0" width="30.0" x="183.0" y="205.0"/>\n          <y:Fill color="#FFCC00" transparent="false"/>\n          <y:BorderStyle color="#000000" type="line" width="1.0"/>\n          <y:NodeLabel alignment="center" autoSizePolicy="content"\n          borderDistance="0.0" fontFamily="Dialog" fontSize="13"\n          fontStyle="plain" hasBackgroundColor="false" hasLineColor="false"\n          height="19.1328125" modelName="internal" modelPosition="c"\n          textColor="#000000" visible="true" width="12.27099609375"\n          x="8.864501953125" y="5.43359375">2</y:NodeLabel>\n          <y:Shape type="rectangle"/>\n        </y:ShapeNode>\n      </data>\n    </node>\n    <node id="n2">\n      <data key="d6" xml:space="preserve"><![CDATA[description\nline1\nline2]]></data>\n      <data key="d3">\n        <y:GenericNode configuration="com.yworks.flowchart.terminator">\n          <y:Geometry height="40.0" width="80.0" x="950.0" y="286.0"/>\n          <y:Fill color="#E8EEF7" color2="#B7C9E3" transparent="false"/>\n          <y:BorderStyle color="#000000" type="line" width="1.0"/>\n          <y:NodeLabel alignment="center" autoSizePolicy="content"\n          fontFamily="Dialog" fontSize="12" fontStyle="plain"\n          hasBackgroundColor="false" hasLineColor="false" height="17.96875"\n          horizontalTextPosition="center" iconTextGap="4" modelName="custom"\n          textColor="#000000" verticalTextPosition="bottom" visible="true"\n          width="67.984375" x="6.0078125" xml:space="preserve"\n          y="11.015625">3<y:LabelModel>\n          <y:SmartNodeLabelModel distance="4.0"/></y:LabelModel>\n          <y:ModelParameter><y:SmartNodeLabelModelParameter labelRatioX="0.0"\n          labelRatioY="0.0" nodeRatioX="0.0" nodeRatioY="0.0" offsetX="0.0"\n          offsetY="0.0" upX="0.0" upY="-1.0"/></y:ModelParameter></y:NodeLabel>\n        </y:GenericNode>\n      </data>\n    </node>\n    <edge id="e0" source="n0" target="n1">\n      <data key="d7">\n        <y:PolyLineEdge>\n          <y:Path sx="0.0" sy="0.0" tx="0.0" ty="0.0"/>\n          <y:LineStyle color="#000000" type="line" width="1.0"/>\n          <y:Arrows source="none" target="standard"/>\n          <y:BendStyle smoothed="false"/>\n        </y:PolyLineEdge>\n      </data>\n    </edge>\n  </graph>\n  <data key="d0">\n    <y:Resources/>\n  </data>\n</graphml>\n'
    fh = io.BytesIO(data.encode('UTF-8'))
    G = nx.read_graphml(fh, force_multigraph=True)
    assert list(G.edges()) == [('n0', 'n1')]
    assert G.has_edge('n0', 'n1', key='e0')
    assert G.nodes['n0']['label'] == '1'
    assert G.nodes['n1']['label'] == '2'
    assert G.nodes['n2']['label'] == '3'
    assert G.nodes['n0']['shape_type'] == 'rectangle'
    assert G.nodes['n1']['shape_type'] == 'rectangle'
    assert G.nodes['n2']['shape_type'] == 'com.yworks.flowchart.terminator'
    assert G.nodes['n2']['description'] == 'description\nline1\nline2'
    fh.seek(0)
    G = nx.read_graphml(fh)
    assert list(G.edges()) == [('n0', 'n1')]
    assert G['n0']['n1']['id'] == 'e0'
    assert G.nodes['n0']['label'] == '1'
    assert G.nodes['n1']['label'] == '2'
    assert G.nodes['n2']['label'] == '3'
    assert G.nodes['n0']['shape_type'] == 'rectangle'
    assert G.nodes['n1']['shape_type'] == 'rectangle'
    assert G.nodes['n2']['shape_type'] == 'com.yworks.flowchart.terminator'
    assert G.nodes['n2']['description'] == 'description\nline1\nline2'
    H = nx.parse_graphml(data, force_multigraph=True)
    assert list(H.edges()) == [('n0', 'n1')]
    assert H.has_edge('n0', 'n1', key='e0')
    assert H.nodes['n0']['label'] == '1'
    assert H.nodes['n1']['label'] == '2'
    assert H.nodes['n2']['label'] == '3'
    H = nx.parse_graphml(data)
    assert list(H.edges()) == [('n0', 'n1')]
    assert H['n0']['n1']['id'] == 'e0'
    assert H.nodes['n0']['label'] == '1'
    assert H.nodes['n1']['label'] == '2'
    assert H.nodes['n2']['label'] == '3'