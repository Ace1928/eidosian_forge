import itertools
import re
import socket
import subprocess
import warnings
from contextlib import contextmanager
from math import pi, log
from typing import (
import httpx
import networkx as nx
import numpy as np
from qcs_api_client.client import QCSClientConfiguration
from qcs_api_client.models import ListQuantumProcessorsResponse
from qcs_api_client.operations.sync import list_quantum_processors
from rpcq.messages import ParameterAref
from pyquil.api import EngagementManager
from pyquil.api._abstract_compiler import AbstractCompiler, QuantumExecutable
from pyquil.api._compiler import QPUCompiler, QVMCompiler
from pyquil.api._qam import QAM, QAMExecutionResult
from pyquil.api._qcs_client import qcs_client
from pyquil.api._qpu import QPU
from pyquil.api._qvm import QVM
from pyquil.experiment._main import Experiment
from pyquil.experiment._memory import merge_memory_map_lists
from pyquil.experiment._result import ExperimentResult, bitstrings_to_expectations
from pyquil.experiment._setting import ExperimentSetting
from pyquil.external.rpcq import CompilerISA
from pyquil.gates import RX, MEASURE
from pyquil.noise import decoherence_noise_with_asymmetric_ro, NoiseModel
from pyquil.paulis import PauliTerm
from pyquil.pyqvm import PyQVM
from pyquil.quantum_processor import (
from pyquil.quil import Program
@contextmanager
def local_forest_runtime(*, host: str='127.0.0.1', qvm_port: int=5000, quilc_port: int=5555, use_protoquil: bool=False) -> Iterator[Tuple[Optional[subprocess.Popen], Optional[subprocess.Popen]]]:
    """A context manager for local QVM and QUIL compiler.

    You must first have installed the `qvm` and `quilc` executables from
    the forest SDK. [https://www.rigetti.com/forest]

    This context manager will ensure that the designated ports are not used, start up `qvm` and
    `quilc` proccesses if possible and terminate them when the context is exited.
    If one of the ports is in use, a ``RuntimeWarning`` will be issued and the `qvm`/`quilc` process
    won't be started.

    .. note::
        Only processes started by this context manager will be terminated on exit, no external
        process will be touched.


    >>> from pyquil import get_qc, Program
    >>> from pyquil.gates import CNOT, Z
    >>> from pyquil.api import local_forest_runtime
    >>>
    >>> qvm = get_qc('9q-square-qvm')
    >>> prog = Program(Z(0), CNOT(0, 1))
    >>>
    >>> with local_forest_runtime():
    >>>     results = qvm.run_and_measure(prog, trials=10)

    :param host: Host on which `qvm` and `quilc` should listen on.
    :param qvm_port: Port which should be used by `qvm`.
    :param quilc_port: Port which should be used by `quilc`.
    :param use_protoquil: Restrict input/output to protoquil.

    .. warning::
        If ``use_protoquil`` is set to ``True`` language features you need
        may be disabled. Please use it with caution.

    :raises: FileNotFoundError: If either executable is not installed.

    :returns: The returned tuple contains two ``subprocess.Popen`` objects
        for the `qvm` and the `quilc` processes.  If one of the designated
        ports is in use, the process won't be started and the respective
        value in the tuple will be ``None``.
    """
    qvm: Optional[subprocess.Popen] = None
    quilc: Optional[subprocess.Popen] = None
    if _port_used(host if host != '0.0.0.0' else '127.0.0.1', qvm_port):
        warning_msg = 'Unable to start qvm server, since the specified port {} is in use.'.format(qvm_port)
        warnings.warn(RuntimeWarning(warning_msg))
    else:
        qvm_cmd = ['qvm', '-S', '--host', host, '-p', str(qvm_port)]
        qvm = subprocess.Popen(qvm_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if _port_used(host if host != '0.0.0.0' else '127.0.0.1', quilc_port):
        warning_msg = 'Unable to start quilc server, since the specified port {} is in use.'.format(quilc_port)
        warnings.warn(RuntimeWarning(warning_msg))
    else:
        quilc_cmd = ['quilc', '--host', host, '-p', str(quilc_port), '-R']
        if use_protoquil:
            quilc_cmd += ['-P']
        quilc = subprocess.Popen(quilc_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        yield (qvm, quilc)
    finally:
        if qvm:
            qvm.terminate()
        if quilc:
            quilc.terminate()