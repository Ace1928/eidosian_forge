from qiskit.circuit import QuantumCircuit, CircuitInstruction, ClassicalRegister
from qiskit.circuit.controlflow import condition_resources
from . import DAGOpNode, DAGCircuit, DAGDependency
from .exceptions import DAGCircuitError
def collect_all_matching_blocks(self, filter_fn, split_blocks=True, min_block_size=2, split_layers=False, collect_from_back=False):
    """Collects all blocks that match a given filtering function filter_fn.
        This iteratively finds the largest block that does not match filter_fn,
        then the largest block that matches filter_fn, and so on, until no more uncollected
        nodes remain. Intuitively, finding larger blocks of non-matching nodes helps to
        find larger blocks of matching nodes later on.

        After the blocks are collected, they can be optionally refined. The option
        ``split_blocks`` allows to split collected blocks into sub-blocks over disjoint
        qubit subsets. The option ``split_layers`` allows to split collected blocks
        into layers of non-overlapping instructions. The option ``min_block_size``
        specifies the minimum number of gates in the block for the block to be collected.

        By default, blocks are collected in the direction from the inputs towards the outputs
        of the circuit. The option ``collect_from_back`` allows to change this direction,
        that is collect blocks from the outputs towards the inputs of the circuit.

        Returns the list of matching blocks only.
        """

    def not_filter_fn(node):
        """Returns the opposite of filter_fn."""
        return not filter_fn(node)
    self._collect_from_back = collect_from_back
    self._setup_in_degrees()
    matching_blocks = []
    while self._have_uncollected_nodes():
        self.collect_matching_block(not_filter_fn)
        matching_block = self.collect_matching_block(filter_fn)
        if matching_block:
            matching_blocks.append(matching_block)
    if split_layers:
        tmp_blocks = []
        for block in matching_blocks:
            tmp_blocks.extend(split_block_into_layers(block))
        matching_blocks = tmp_blocks
    if split_blocks:
        tmp_blocks = []
        for block in matching_blocks:
            tmp_blocks.extend(BlockSplitter().run(block))
        matching_blocks = tmp_blocks
    if self._collect_from_back:
        matching_blocks = [block[::-1] for block in matching_blocks[::-1]]
    matching_blocks = [block for block in matching_blocks if len(block) >= min_block_size]
    return matching_blocks