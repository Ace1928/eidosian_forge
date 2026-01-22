import warnings
from collections import defaultdict
import networkx as nx
from networkx.utils import open_file
def make_graph(self, graph_xml, graphml_keys, defaults, G=None):
    edgedefault = graph_xml.get('edgedefault', None)
    if G is None:
        if edgedefault == 'directed':
            G = nx.MultiDiGraph()
        else:
            G = nx.MultiGraph()
    G.graph['node_default'] = {}
    G.graph['edge_default'] = {}
    for key_id, value in defaults.items():
        key_for = graphml_keys[key_id]['for']
        name = graphml_keys[key_id]['name']
        python_type = graphml_keys[key_id]['type']
        if key_for == 'node':
            G.graph['node_default'].update({name: python_type(value)})
        if key_for == 'edge':
            G.graph['edge_default'].update({name: python_type(value)})
    hyperedge = graph_xml.find(f'{{{self.NS_GRAPHML}}}hyperedge')
    if hyperedge is not None:
        raise nx.NetworkXError("GraphML reader doesn't support hyperedges")
    for node_xml in graph_xml.findall(f'{{{self.NS_GRAPHML}}}node'):
        self.add_node(G, node_xml, graphml_keys, defaults)
    for edge_xml in graph_xml.findall(f'{{{self.NS_GRAPHML}}}edge'):
        self.add_edge(G, edge_xml, graphml_keys)
    data = self.decode_data_elements(graphml_keys, graph_xml)
    G.graph.update(data)
    if self.multigraph:
        return G
    G = nx.DiGraph(G) if G.is_directed() else nx.Graph(G)
    nx.set_edge_attributes(G, values=self.edge_ids, name='id')
    return G