from typing import Iterable, Union
import networkx as nx
import rustworkx as rx
import pennylane as qml
from pennylane import qaoa
def edge_driver(graph: Union[nx.Graph, rx.PyGraph], reward: list):
    """Returns the edge-driver cost Hamiltonian.

    Given some graph, :math:`G` with each node representing a wire, and a binary
    colouring where each node/wire is assigned either :math:`|0\\rangle` or :math:`|1\\rangle`, the edge driver
    cost Hamiltonian will assign a lower energy to edges represented by qubit states with endpoint colourings
    supplied in ``reward``.

    For instance, if ``reward`` is ``["11"]``, then edges
    with both endpoints coloured as ``1`` (the state :math:`|11\\rangle`) will be assigned a lower energy, while
    the other colourings  (``"00"``, ``"10"``, and ``"01"`` corresponding to states
    :math:`|00\\rangle`, :math:`|10\\rangle`, and :math:`|10\\rangle`, respectively) will be assigned a higher energy.

    See usage details for more information.

    Args:
         graph (nx.Graph or rx.PyGraph): The graph on which the Hamiltonian is defined
         reward (list[str]): The list of two-bit bitstrings that are assigned a lower energy by the Hamiltonian

    Returns:
        .Hamiltonian:

    **Example**

    >>> import networkx as nx
    >>> graph = nx.Graph([(0, 1), (1, 2)])
    >>> hamiltonian = qaoa.edge_driver(graph, ["11", "10", "01"])
    >>> print(hamiltonian)
      (0.25) [Z0]
    + (0.25) [Z1]
    + (0.25) [Z1]
    + (0.25) [Z2]
    + (0.25) [Z0 Z1]
    + (0.25) [Z1 Z2]

    >>> import rustworkx as rx
    >>> graph = rx.PyGraph()
    >>> graph.add_nodes_from([0, 1, 2])
    >>> graph.add_edges_from([(0, 1,""), (1,2,"")])
    >>> hamiltonian = qaoa.edge_driver(graph, ["11", "10", "01"])
    >>> print(hamiltonian)
      (0.25) [Z0]
    + (0.25) [Z1]
    + (0.25) [Z1]
    + (0.25) [Z2]
    + (0.25) [Z0 Z1]
    + (0.25) [Z1 Z2]

    In the above example, ``"11"``, ``"10"``, and ``"01"`` are assigned a lower
    energy than ``"00"``. For example, a quick calculation of expectation values gives us:

    .. math:: \\langle 000 | H | 000 \\rangle \\ = \\ 1.5
    .. math:: \\langle 100 | H | 100 \\rangle \\ = \\ 0.5
    .. math:: \\langle 110 | H | 110\\rangle \\ = \\ -0.5

    In the first example, both vertex pairs are not in ``reward``. In the second example, one pair is in ``reward`` and
    the other is not. Finally, in the third example, both pairs are in ``reward``.

    .. details::
        :title: Usage Details

        The goal of many combinatorial problems that can be solved with QAOA is to
        find a `Graph colouring <https://en.wikipedia.org/wiki/Graph_coloring>`__ of some supplied
        graph :math:`G`, that minimizes some cost function. With QAOA, it is natural to consider the class
        of graph colouring problems that only admit two colours, as we can easily encode these two colours
        using the :math:`|1\\rangle` and :math:`|0\\rangle` states of qubits. Therefore, given
        some graph :math:`G`, each edge of the graph can be described by a pair of qubits, :math:`|00\\rangle`,
        :math:`|01\\rangle`, :math:`|10\\rangle`, or :math:`|11\\rangle`, corresponding to the colourings of its endpoints.

        When constructing QAOA cost functions, one must "penalize" certain states of the graph, and "reward"
        others, by assigning higher and lower energies to these respective configurations. Given a set of vertex-colour
        pairs (which each describe a possible  state of a graph edge), the ``edge_driver()``
        function outputs a Hamiltonian that rewards the pairs in the set, and penalizes the others.

        For example, given the reward set: :math:`\\{|00\\rangle, \\ |01\\rangle, \\ |10\\rangle\\}` and the graph :math:`G`,
        the ``edge_driver()`` function will output the following Hamiltonian:

        .. math:: H \\ = \\ \\frac{1}{4} \\displaystyle\\sum_{(i, j) \\in E(G)} \\big( Z_{i} Z_{j} \\ - \\ Z_{i} \\ - \\ Z_{j} \\big)

        where :math:`E(G)` is the set of edges of :math:`G`, and :math:`Z_i` is the Pauli-Z operator acting on the
        :math:`i`-th wire. As can be checked, this Hamiltonian assigns an energy of :math:`-1/4` to the states
        :math:`|00\\rangle`, :math:`|01\\rangle` and :math:`|10\\rangle`, and an energy of :math:`3/4` to the state
        :math:`|11\\rangle`.

        .. Note::

            ``reward`` must always contain both :math:`|01\\rangle` and :math:`|10\\rangle`, or neither of the two.
            Within an undirected graph, there is no notion of "order"
            of edge endpoints, so these two states are effectively the same. Therefore, there is no well-defined way to
            penalize one and reward the other.

        .. Note::

            The absolute difference in energy between colourings in ``reward`` and colourings in its
            complement is always :math:`1`.

    """
    allowed = ['00', '01', '10', '11']
    if not all((e in allowed for e in reward)):
        raise ValueError("Encountered invalid entry in 'reward', expected 2-bit bitstrings.")
    if '01' in reward and '10' not in reward or ('10' in reward and '01' not in reward):
        raise ValueError("'reward' cannot contain either '10' or '01', must contain neither or both.")
    if not isinstance(graph, (nx.Graph, rx.PyGraph)):
        raise ValueError(f'Input graph must be a nx.Graph or rx.PyGraph, got {type(graph).__name__}')
    coeffs = []
    ops = []
    is_rx = isinstance(graph, rx.PyGraph)
    graph_nodes = graph.nodes()
    graph_edges = sorted(graph.edge_list()) if is_rx else graph.edges
    get_nvalue = lambda i: graph_nodes[i] if is_rx else i
    if len(reward) == 0 or len(reward) == 4:
        coeffs = [1 for _ in graph_nodes]
        ops = [qml.Identity(v) for v in graph_nodes]
    else:
        reward = list(set(reward) - {'01'})
        sign = -1
        if len(reward) == 2:
            reward = list({'00', '10', '11'} - set(reward))
            sign = 1
        reward = reward[0]
        if reward == '00':
            for e in graph_edges:
                coeffs.extend([0.25 * sign, 0.25 * sign, 0.25 * sign])
                ops.extend([qml.Z(get_nvalue(e[0])) @ qml.Z(get_nvalue(e[1])), qml.Z(get_nvalue(e[0])), qml.Z(get_nvalue(e[1]))])
        if reward == '10':
            for e in graph_edges:
                coeffs.append(-0.5 * sign)
                ops.append(qml.Z(get_nvalue(e[0])) @ qml.Z(get_nvalue(e[1])))
        if reward == '11':
            for e in graph_edges:
                coeffs.extend([0.25 * sign, -0.25 * sign, -0.25 * sign])
                ops.extend([qml.Z(get_nvalue(e[0])) @ qml.Z(get_nvalue(e[1])), qml.Z(get_nvalue(e[0])), qml.Z(get_nvalue(e[1]))])
    return qml.Hamiltonian(coeffs, ops)