from qiskit.transpiler.basepasses import TransformationPass
from qiskit.converters import dag_to_dagdependency, dagdependency_to_dag
from qiskit.dagcircuit.collect_blocks import BlockCollector, BlockCollapser
from qiskit.transpiler.passes.utils import control_flow
def collect_using_filter_function(dag, filter_function, split_blocks, min_block_size, split_layers=False, collect_from_back=False):
    """Corresponds to an important block collection strategy that greedily collects
    maximal blocks of nodes matching a given ``filter_function``.
    """
    return BlockCollector(dag).collect_all_matching_blocks(filter_fn=filter_function, split_blocks=split_blocks, min_block_size=min_block_size, split_layers=split_layers, collect_from_back=collect_from_back)