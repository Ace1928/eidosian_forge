from collections import defaultdict
import networkx as nx
def is_on_outer_face(x):
    return x not in marked_nodes and (x in outer_face_ccw_nbr or x == v1)