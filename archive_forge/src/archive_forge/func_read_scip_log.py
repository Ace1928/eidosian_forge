import os
import subprocess
from pyomo.common import Executable
from pyomo.common.collections import Bunch
from pyomo.common.tempfiles import TempfileManager
from pyomo.opt.base import ProblemFormat, ResultsFormat
from pyomo.opt.base.solvers import _extract_version, SolverFactory
from pyomo.opt.results import (
from pyomo.opt.solver import SystemCallSolver
import logging
@staticmethod
def read_scip_log(filename: str):
    from collections import deque
    with open(filename) as f:
        scip_lines = list(deque(f, 7))
        scip_lines.pop()
    expected_labels = ['SCIP Status        :', 'Solving Time (sec) :', 'Solving Nodes      :', 'Primal Bound       :', 'Dual Bound         :', 'Gap                :']
    colon_position = 19
    for i, log_file_line in enumerate(scip_lines):
        if expected_labels[i] != log_file_line[0:colon_position + 1]:
            return {}
    solver_status = scip_lines[0][colon_position + 2:scip_lines[0].index('\n')]
    solving_time = float(scip_lines[1][colon_position + 2:scip_lines[1].index('\n')])
    try:
        solving_nodes = int(scip_lines[2][colon_position + 2:scip_lines[2].index('(')])
    except ValueError:
        solving_nodes = int(scip_lines[2][colon_position + 2:scip_lines[2].index('\n')])
    primal_bound = float(scip_lines[3][colon_position + 2:scip_lines[3].index('(')])
    dual_bound = float(scip_lines[4][colon_position + 2:scip_lines[4].index('\n')])
    try:
        gap = float(scip_lines[5][colon_position + 2:scip_lines[5].index('%')])
    except ValueError:
        gap = scip_lines[5][colon_position + 2:scip_lines[5].index('\n')]
        if gap == 'infinite':
            gap = float('inf')
    out_dict = {'solver_status': solver_status, 'solving_time': solving_time, 'solving_nodes': solving_nodes, 'primal_bound': primal_bound, 'dual_bound': dual_bound, 'gap': gap}
    return out_dict