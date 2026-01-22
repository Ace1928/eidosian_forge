import warnings
import networkx as nx
from networkx.utils import open_file
@nx._dispatch(graphs=None)
def parse_pajek(lines):
    """Parse Pajek format graph from string or iterable.

    Parameters
    ----------
    lines : string or iterable
       Data in Pajek format.

    Returns
    -------
    G : NetworkX graph

    See Also
    --------
    read_pajek

    """
    import shlex
    if isinstance(lines, str):
        lines = iter(lines.split('\n'))
    lines = iter([line.rstrip('\n') for line in lines])
    G = nx.MultiDiGraph()
    labels = []
    while lines:
        try:
            l = next(lines)
        except:
            break
        if l.lower().startswith('*network'):
            try:
                label, name = l.split(None, 1)
            except ValueError:
                pass
            else:
                G.graph['name'] = name
        elif l.lower().startswith('*vertices'):
            nodelabels = {}
            l, nnodes = l.split()
            for i in range(int(nnodes)):
                l = next(lines)
                try:
                    splitline = [x.decode('utf-8') for x in shlex.split(str(l).encode('utf-8'))]
                except AttributeError:
                    splitline = shlex.split(str(l))
                id, label = splitline[0:2]
                labels.append(label)
                G.add_node(label)
                nodelabels[id] = label
                G.nodes[label]['id'] = id
                try:
                    x, y, shape = splitline[2:5]
                    G.nodes[label].update({'x': float(x), 'y': float(y), 'shape': shape})
                except:
                    pass
                extra_attr = zip(splitline[5::2], splitline[6::2])
                G.nodes[label].update(extra_attr)
        elif l.lower().startswith('*edges') or l.lower().startswith('*arcs'):
            if l.lower().startswith('*edge'):
                G = nx.MultiGraph(G)
            if l.lower().startswith('*arcs'):
                G = G.to_directed()
            for l in lines:
                try:
                    splitline = [x.decode('utf-8') for x in shlex.split(str(l).encode('utf-8'))]
                except AttributeError:
                    splitline = shlex.split(str(l))
                if len(splitline) < 2:
                    continue
                ui, vi = splitline[0:2]
                u = nodelabels.get(ui, ui)
                v = nodelabels.get(vi, vi)
                edge_data = {}
                try:
                    w = splitline[2:3]
                    edge_data.update({'weight': float(w[0])})
                except:
                    pass
                extra_attr = zip(splitline[3::2], splitline[4::2])
                edge_data.update(extra_attr)
                G.add_edge(u, v, **edge_data)
        elif l.lower().startswith('*matrix'):
            G = nx.DiGraph(G)
            adj_list = ((labels[row], labels[col], {'weight': int(data)}) for row, line in enumerate(lines) for col, data in enumerate(line.split()) if int(data) != 0)
            G.add_edges_from(adj_list)
    return G