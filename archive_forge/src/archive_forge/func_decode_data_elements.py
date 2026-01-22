import warnings
from collections import defaultdict
import networkx as nx
from networkx.utils import open_file
def decode_data_elements(self, graphml_keys, obj_xml):
    """Use the key information to decode the data XML if present."""
    data = {}
    for data_element in obj_xml.findall(f'{{{self.NS_GRAPHML}}}data'):
        key = data_element.get('key')
        try:
            data_name = graphml_keys[key]['name']
            data_type = graphml_keys[key]['type']
        except KeyError as err:
            raise nx.NetworkXError(f'Bad GraphML data: no key {key}') from err
        text = data_element.text
        if text is not None and len(list(data_element)) == 0:
            if data_type == bool:
                data[data_name] = self.convert_bool[text.lower()]
            else:
                data[data_name] = data_type(text)
        elif len(list(data_element)) > 0:
            node_label = None
            gn = data_element.find(f'{{{self.NS_Y}}}GenericNode')
            if gn:
                data['shape_type'] = gn.get('configuration')
            for node_type in ['GenericNode', 'ShapeNode', 'SVGNode', 'ImageNode']:
                pref = f'{{{self.NS_Y}}}{node_type}/{{{self.NS_Y}}}'
                geometry = data_element.find(f'{pref}Geometry')
                if geometry is not None:
                    data['x'] = geometry.get('x')
                    data['y'] = geometry.get('y')
                if node_label is None:
                    node_label = data_element.find(f'{pref}NodeLabel')
                shape = data_element.find(f'{pref}Shape')
                if shape is not None:
                    data['shape_type'] = shape.get('type')
            if node_label is not None:
                data['label'] = node_label.text
            for edge_type in ['PolyLineEdge', 'SplineEdge', 'QuadCurveEdge', 'BezierEdge', 'ArcEdge']:
                pref = f'{{{self.NS_Y}}}{edge_type}/{{{self.NS_Y}}}'
                edge_label = data_element.find(f'{pref}EdgeLabel')
                if edge_label is not None:
                    break
            if edge_label is not None:
                data['label'] = edge_label.text
    return data