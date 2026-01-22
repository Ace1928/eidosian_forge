import collections
import copy
import itertools
from qiskit.circuit import Parameter, ParameterExpression
from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit.dagcircuit.dagdependency import DAGDependency
from qiskit.converters.dagdependency_to_dag import dagdependency_to_dag
from qiskit.utils import optionals as _optionals
def run_dag_opt(self):
    """
        It runs the substitution algorithm and creates the optimized DAGCircuit().
        """
    self._substitution()
    dag_dep_opt = DAGDependency()
    dag_dep_opt.name = self.circuit_dag_dep.name
    qregs = list(self.circuit_dag_dep.qregs.values())
    cregs = list(self.circuit_dag_dep.cregs.values())
    for register in qregs:
        dag_dep_opt.add_qreg(register)
    for register in cregs:
        dag_dep_opt.add_creg(register)
    already_sub = []
    if self.substitution_list:
        for group in self.substitution_list:
            circuit_sub = group.circuit_config
            template_inverse = group.template_config
            pred = group.pred_block
            qubit = group.qubit_config[0]
            if group.clbit_config:
                clbit = group.clbit_config[0]
            else:
                clbit = []
            for elem in pred:
                node = self.circuit_dag_dep.get_node(elem)
                inst = node.op.copy()
                dag_dep_opt.add_op_node(inst, node.qargs, node.cargs)
                already_sub.append(elem)
            already_sub = already_sub + circuit_sub
            for index in template_inverse:
                all_qubits = self.circuit_dag_dep.qubits
                qarg_t = group.template_dag_dep.get_node(index).qindices
                qarg_c = [qubit[x] for x in qarg_t]
                qargs = [all_qubits[x] for x in qarg_c]
                all_clbits = self.circuit_dag_dep.clbits
                carg_t = group.template_dag_dep.get_node(index).cindices
                if all_clbits and clbit:
                    carg_c = [clbit[x] for x in carg_t]
                    cargs = [all_clbits[x] for x in carg_c]
                else:
                    cargs = []
                node = group.template_dag_dep.get_node(index)
                inst = node.op.copy()
                dag_dep_opt.add_op_node(inst.inverse(), qargs, cargs)
        for node_id in self.unmatched_list:
            node = self.circuit_dag_dep.get_node(node_id)
            inst = node.op.copy()
            dag_dep_opt.add_op_node(inst, node.qargs, node.cargs)
        dag_dep_opt._add_predecessors()
        dag_dep_opt._add_successors()
    else:
        dag_dep_opt = self.circuit_dag_dep
    self.dag_dep_optimized = dag_dep_opt
    self.dag_optimized = dagdependency_to_dag(dag_dep_opt)