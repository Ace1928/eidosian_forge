from pyomo.contrib.pynumero.interfaces.nlp import NLP
from pyomo.contrib.pynumero.sparse import BlockVector, BlockMatrix
from pyomo.contrib.pynumero.linalg.ma27_interface import MA27
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
import numpy as np
from scipy.sparse import tril
import pyomo.environ as pe
from pyomo import dae
from pyomo.common.timing import TicTocTimer
import time
from pyomo.contrib.pynumero.linalg.base import LinearSolverInterface, LinearSolverStatus
def load_solution(m: pe.ConcreteModel(), nlp: PyomoNLP):
    primals = nlp.get_primals()
    pyomo_vars = nlp.get_pyomo_variables()
    for v, val in zip(pyomo_vars, primals):
        v.value = val