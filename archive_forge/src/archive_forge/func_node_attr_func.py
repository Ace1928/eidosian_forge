from rustworkx.visualization import graphviz_draw
from qiskit.dagcircuit.dagnode import DAGOpNode, DAGInNode, DAGOutNode
from qiskit.circuit import Qubit, Clbit, ClassicalRegister
from qiskit.circuit.classical import expr
from qiskit.converters import dagdependency_to_circuit
from qiskit.utils import optionals as _optionals
from qiskit.exceptions import InvalidFileError
from .exceptions import VisualizationError
def node_attr_func(node):
    if style == 'plain':
        return {}
    if style == 'color':
        n = {}
        if isinstance(node, DAGOpNode):
            n['label'] = node.name
            n['color'] = 'blue'
            n['style'] = 'filled'
            n['fillcolor'] = 'lightblue'
        if isinstance(node, DAGInNode):
            if isinstance(node.wire, Qubit):
                label = register_bit_labels.get(node.wire, f'q_{dag.find_bit(node.wire).index}')
            else:
                label = register_bit_labels.get(node.wire, f'c_{dag.find_bit(node.wire).index}')
            n['label'] = label
            n['color'] = 'black'
            n['style'] = 'filled'
            n['fillcolor'] = 'green'
        if isinstance(node, DAGOutNode):
            if isinstance(node.wire, Qubit):
                label = register_bit_labels.get(node.wire, f'q[{dag.find_bit(node.wire).index}]')
            else:
                label = register_bit_labels.get(node.wire, f'c[{dag.find_bit(node.wire).index}]')
            n['label'] = label
            n['color'] = 'black'
            n['style'] = 'filled'
            n['fillcolor'] = 'red'
        return n
    else:
        raise VisualizationError('Invalid style %s' % style)