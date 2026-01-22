import os
from abc import ABC, abstractmethod
import torch.testing._internal.dist_utils
def setup_fault_injection(self, faulty_messages, messages_to_delay):
    """Method used by dist_init to prepare the faulty agent.

        Does nothing for other agents.
        """
    pass