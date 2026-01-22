from multiprocessing import Process
import pytest
import cirq
import cirq.contrib.routing as ccr
from cirq.contrib.routing.greedy import route_circuit_greedily
Run a separate process and check if greedy router hits timeout (5s).