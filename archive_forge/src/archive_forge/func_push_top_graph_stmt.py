from pyparsing import (
import pydot
def push_top_graph_stmt(s, loc, toks):
    attrs = {}
    g = None
    for element in toks:
        if isinstance(element, (ParseResults, tuple, list)) and len(element) == 1 and isinstance(element[0], str):
            element = element[0]
        if element == 'strict':
            attrs['strict'] = True
        elif element in ['graph', 'digraph']:
            attrs = {}
            g = pydot.Dot(graph_type=element, **attrs)
            attrs['type'] = element
            top_graphs.append(g)
        elif isinstance(element, str):
            g.set_name(element)
        elif isinstance(element, pydot.Subgraph):
            g.obj_dict['attributes'].update(element.obj_dict['attributes'])
            g.obj_dict['edges'].update(element.obj_dict['edges'])
            g.obj_dict['nodes'].update(element.obj_dict['nodes'])
            g.obj_dict['subgraphs'].update(element.obj_dict['subgraphs'])
            g.set_parent_graph(g)
        elif isinstance(element, P_AttrList):
            attrs.update(element.attrs)
        elif isinstance(element, (ParseResults, list)):
            add_elements(g, element)
        else:
            raise ValueError('Unknown element statement: {s}'.format(s=element))
    for g in top_graphs:
        update_parent_graph_hierarchy(g)
    if len(top_graphs) == 1:
        return top_graphs[0]
    return top_graphs