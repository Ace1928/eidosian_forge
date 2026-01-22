from __future__ import annotations
import copy
import logging
import os.path
import subprocess
import warnings
from collections import defaultdict, namedtuple
from itertools import combinations
from operator import itemgetter
from shutil import which
from typing import TYPE_CHECKING, Any, Callable, cast
import networkx as nx
import networkx.algorithms.isomorphism as iso
import numpy as np
from monty.dev import deprecated
from monty.json import MSONable
from networkx.drawing.nx_agraph import write_dot
from networkx.readwrite import json_graph
from scipy.spatial import KDTree
from scipy.stats import describe
from pymatgen.core import Lattice, Molecule, PeriodicSite, Structure
from pymatgen.core.structure import FunctionalGroups
from pymatgen.util.coord import lattice_points_in_supercell
from pymatgen.vis.structure_vtk import EL_COLORS
def draw_graph_to_file(self, filename='graph', diff=None, hide_unconnected_nodes=False, hide_image_edges=True, edge_colors=False, node_labels=False, weight_labels=False, image_labels=False, color_scheme='VESTA', keep_dot=False, algo='fdp'):
    """
        Draws graph using GraphViz.

        The networkx graph object itself can also be drawn
        with networkx's in-built graph drawing methods, but
        note that this might give misleading results for
        multigraphs (edges are super-imposed on each other).

        If visualization is difficult to interpret,
        `hide_image_edges` can help, especially in larger
        graphs.

        Args:
            filename: filename to output, will detect filetype
                from extension (any graphviz filetype supported, such as
                pdf or png)
            diff (StructureGraph): an additional graph to
                compare with, will color edges red that do not exist in diff
                and edges green that are in diff graph but not in the
                reference graph
            hide_unconnected_nodes: if True, hide unconnected nodes
            hide_image_edges: if True, do not draw edges that
                go through periodic boundaries
            edge_colors (bool): if True, use node colors to color edges
            node_labels (bool): if True, label nodes with
                species and site index
            weight_labels (bool): if True, label edges with weights
            image_labels (bool): if True, label edges with
                their periodic images (usually only used for debugging,
                edges to periodic images always appear as dashed lines)
            color_scheme (str): "VESTA" or "JMOL"
            keep_dot (bool): keep GraphViz .dot file for later visualization
            algo: any graphviz algo, "neato" (for simple graphs)
                or "fdp" (for more crowded graphs) usually give good outputs
        """
    if not which(algo):
        raise RuntimeError('StructureGraph graph drawing requires GraphViz binaries to be in the path.')
    g = self.graph.copy()
    g.graph = {'nodesep': 10.0, 'dpi': 300, 'overlap': 'false'}
    for node in g.nodes():
        label = f'{self.molecule[node].specie}({node})' if node_labels else ''
        c = EL_COLORS[color_scheme].get(str(self.molecule[node].specie.symbol), [0, 0, 0])
        fontcolor = '#000000' if 1 - (c[0] * 0.299 + c[1] * 0.587 + c[2] * 0.114) / 255 < 0.5 else '#ffffff'
        color = f'#{c[0]:02x}{c[1]:02x}{c[2]:02x}'
        g.add_node(node, fillcolor=color, fontcolor=fontcolor, label=label, fontname='Helvetica-bold', style='filled', shape='circle')
    edges_to_delete = []
    for u, v, k, d in g.edges(keys=True, data=True):
        to_image = d['to_jimage'] if 'to_image' in d else (0, 0, 0)
        d['style'] = 'solid'
        if to_image != (0, 0, 0):
            d['style'] = 'dashed'
            if hide_image_edges:
                edges_to_delete.append((u, v, k))
        d['arrowhead'] = 'none'
        if image_labels:
            d['headlabel'] = '' if to_image == (0, 0, 0) else f'to {to_image}'
            d['arrowhead'] = 'normal' if d['headlabel'] else 'none'
        color_u = g.nodes[u]['fillcolor']
        color_v = g.nodes[v]['fillcolor']
        d['color_uv'] = f'{color_u};0.5:{color_v};0.5' if edge_colors else '#000000'
        if weight_labels:
            units = g.graph.get('edge_weight_units', '')
            if d.get('weight'):
                d['label'] = f'{d['weight']:.2f} {units}'
        g.edges[u, v, k].update(d)
    if hide_image_edges:
        for edge_to_delete in edges_to_delete:
            g.remove_edge(*edge_to_delete)
    if hide_unconnected_nodes:
        g = g.subgraph([n for n in g.degree() if g.degree()[n] != 0])
    if diff:
        diff = self.diff(diff, strict=True)
        green_edges = []
        red_edges = []
        for u, v, k, d in g.edges(keys=True, data=True):
            if (u, v, d['to_jimage']) in diff['self']:
                red_edges.append((u, v, k))
            elif (u, v, d['to_jimage']) in diff['other']:
                green_edges.append((u, v, k))
        for u, v, k in green_edges:
            g.edges[u, v, k]['color_uv'] = '#00ff00'
        for u, v, k in red_edges:
            g.edges[u, v, k]['color_uv'] = '#ff0000'
    basename, extension = os.path.splitext(filename)
    extension = extension[1:]
    write_dot(g, f'{basename}.dot')
    with open(filename, mode='w') as file:
        args = [algo, '-T', extension, f'{basename}.dot']
        with subprocess.Popen(args, stdout=file, stdin=subprocess.PIPE, close_fds=True) as rs:
            rs.communicate()
            if rs.returncode != 0:
                raise RuntimeError(f'{algo} exited with return code {rs.returncode}.')
    if not keep_dot:
        os.remove(f'{basename}.dot')