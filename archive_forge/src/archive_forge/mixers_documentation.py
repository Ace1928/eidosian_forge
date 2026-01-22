import itertools
import functools
from typing import Iterable, Union
import networkx as nx
import rustworkx as rx
import pennylane as qml
from pennylane.wires import Wires
Creates a bit-flip mixer Hamiltonian.

    This mixer is defined as:

    .. math:: H_M \ = \ \displaystyle\sum_{v \in V(G)} \frac{1}{2^{d(v)}} X_{v}
              \displaystyle\prod_{w \in N(v)} (\mathbb{I} \ + \ (-1)^b Z_w)

    where :math:`V(G)` is the set of vertices of some graph :math:`G`, :math:`d(v)` is the
    `degree <https://en.wikipedia.org/wiki/Degree_(graph_theory)>`__ of vertex :math:`v`, and
    :math:`N(v)` is the `neighbourhood <https://en.wikipedia.org/wiki/Neighbourhood_(graph_theory)>`__
    of vertex :math:`v`. In addition, :math:`Z_v` and :math:`X_v`
    are the Pauli-Z and Pauli-X operators on vertex :math:`v`, respectively,
    and :math:`\mathbb{I}` is the identity operator.

    This mixer was introduced in `Hadfield et al. (2019) <https://doi.org/10.3390/a12020034>`__.

    Args:
         graph (nx.Graph or rx.PyGraph): A graph defining the collections of wires on which the Hamiltonian acts.
         b (int): Either :math:`0` or :math:`1`. When :math:`b=0`, a bit flip is performed on
             vertex :math:`v` only when all neighbouring nodes are in state :math:`|0\rangle`.
             Alternatively, for :math:`b=1`, a bit flip is performed only when all the neighbours of
             :math:`v` are in the state :math:`|1\rangle`.

    Returns:
        Hamiltonian: Mixer Hamiltonian

    **Example**

    The mixer Hamiltonian can be called as follows:

    >>> from pennylane import qaoa
    >>> from networkx import Graph
    >>> graph = Graph([(0, 1), (1, 2)])
    >>> mixer_h = qaoa.bit_flip_mixer(graph, 0)
    >>> print(mixer_h)
      (0.25) [X1]
    + (0.5) [X0]
    + (0.5) [X2]
    + (0.25) [X1 Z2]
    + (0.25) [X1 Z0]
    + (0.5) [X0 Z1]
    + (0.5) [X2 Z1]
    + (0.25) [X1 Z0 Z2]

    >>> import rustworkx as rx
    >>> graph = rx.PyGraph()
    >>> graph.add_nodes_from([0, 1, 2])
    >>> graph.add_edges_from([(0, 1, ""), (1, 2, "")])
    >>> mixer_h = qaoa.bit_flip_mixer(graph, 0)
    >>> print(mixer_h)
      (0.25) [X1]
    + (0.5) [X0]
    + (0.5) [X2]
    + (0.25) [X1 Z0]
    + (0.25) [X1 Z2]
    + (0.5) [X0 Z1]
    + (0.5) [X2 Z1]
    + (0.25) [X1 Z2 Z0]
    