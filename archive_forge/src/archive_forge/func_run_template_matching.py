import itertools
from qiskit.circuit.controlledgate import ControlledGate
from qiskit.transpiler.passes.optimization.template_matching.forward_match import ForwardMatch
from qiskit.transpiler.passes.optimization.template_matching.backward_match import BackwardMatch
def run_template_matching(self):
    """
        Run the complete algorithm for finding all maximal matches for the given template and
        circuit. First it fixes the configuration of the circuit due to the first match.
        Then it explores all compatible qubit configurations of the circuit. For each
        qubit configurations, we apply first the Forward part of the algorithm  and then
        the Backward part of the algorithm. The longest matches for the given configuration
        are stored. Finally, the list of stored matches is sorted.
        """
    n_qubits_c = len(self.circuit_dag_dep.qubits)
    n_clbits_c = len(self.circuit_dag_dep.clbits)
    n_qubits_t = len(self.template_dag_dep.qubits)
    n_clbits_t = len(self.template_dag_dep.clbits)
    for template_index in range(0, self.template_dag_dep.size()):
        for circuit_index in range(0, self.circuit_dag_dep.size()):
            if self.circuit_dag_dep.get_node(circuit_index).op.soft_compare(self.template_dag_dep.get_node(template_index).op):
                qarg_c = self.circuit_dag_dep.get_node(circuit_index).qindices
                carg_c = self.circuit_dag_dep.get_node(circuit_index).cindices
                qarg_t = self.template_dag_dep.get_node(template_index).qindices
                carg_t = self.template_dag_dep.get_node(template_index).cindices
                node_id_c = circuit_index
                node_id_t = template_index
                all_list_first_match_q, list_first_match_c = self._list_first_match_new(self.circuit_dag_dep.get_node(circuit_index), self.template_dag_dep.get_node(template_index), n_qubits_t, n_clbits_t)
                list_circuit_q = list(range(0, n_qubits_c))
                list_circuit_c = list(range(0, n_clbits_c))
                if self.heuristics_qubits_param:
                    heuristics_qubits = self._explore_circuit(node_id_c, node_id_t, n_qubits_t, self.heuristics_qubits_param[0])
                else:
                    heuristics_qubits = []
                for sub_q in self._sublist(list_circuit_q, qarg_c, n_qubits_t - len(qarg_t)):
                    if set(heuristics_qubits).issubset(set(sub_q) | set(qarg_c)):
                        for perm_q in itertools.permutations(sub_q):
                            perm_q = list(perm_q)
                            for list_first_match_q in all_list_first_match_q:
                                list_qubit_circuit = self._list_qubit_clbit_circuit(list_first_match_q, perm_q)
                                if list_circuit_c:
                                    for sub_c in self._sublist(list_circuit_c, carg_c, n_clbits_t - len(carg_t)):
                                        for perm_c in itertools.permutations(sub_c):
                                            perm_c = list(perm_c)
                                            list_clbit_circuit = self._list_qubit_clbit_circuit(list_first_match_c, perm_c)
                                            forward = ForwardMatch(self.circuit_dag_dep, self.template_dag_dep, node_id_c, node_id_t, list_qubit_circuit, list_clbit_circuit)
                                            forward.run_forward_match()
                                            backward = BackwardMatch(forward.circuit_dag_dep, forward.template_dag_dep, forward.match, node_id_c, node_id_t, list_qubit_circuit, list_clbit_circuit, self.heuristics_backward_param)
                                            backward.run_backward_match()
                                            self._add_match(backward.match_final)
                                else:
                                    forward = ForwardMatch(self.circuit_dag_dep, self.template_dag_dep, node_id_c, node_id_t, list_qubit_circuit)
                                    forward.run_forward_match()
                                    backward = BackwardMatch(forward.circuit_dag_dep, forward.template_dag_dep, forward.match, node_id_c, node_id_t, list_qubit_circuit, [], self.heuristics_backward_param)
                                    backward.run_backward_match()
                                    self._add_match(backward.match_final)
    self.match_list.sort(key=lambda x: len(x.match), reverse=True)