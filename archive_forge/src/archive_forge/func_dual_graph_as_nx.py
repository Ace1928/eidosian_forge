from .links import Link, Strand, Crossing, CrossingStrand
from .ordered_set import OrderedSet
from .. import graphs
import random
import networkx as nx
import collections
def dual_graph_as_nx(link):
    corners = OrderedSet([CrossingStrand(c, i) for c in link.crossings for i in range(4)])
    faces = []
    to_face_index = {}
    while len(corners):
        count = len(faces)
        first_cs = corners.pop()
        to_face_index[first_cs] = count
        face = [first_cs]
        while True:
            next = face[-1].next_corner()
            if next == face[0]:
                faces.append(Face(face, count))
                break
            else:
                to_face_index[next] = count
                corners.remove(next)
                face.append(next)
    G = nx.Graph()
    to_face = {edge: faces[f] for edge, f in to_face_index.items()}
    for edge, face in to_face.items():
        opp_edge = edge.opposite()
        neighbor = to_face[opp_edge]
        if face.label < neighbor.label:
            G.add_edge(face.label, neighbor.label, interface={face.label: edge, neighbor.label: opp_edge})
    G.edge_to_face = to_face_index
    return G