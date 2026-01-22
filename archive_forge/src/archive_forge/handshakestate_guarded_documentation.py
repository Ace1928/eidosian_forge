from dissononce.extras.processing.handshakestate_forwarder import ForwarderHandshakeState
from transitions import Machine
from transitions.core import MachineError
import logging

        :param machine_error:
        :type machine_error: MachineError
        :param bad_method:
        :type bad_method: str
        :return:
        :rtype:
        