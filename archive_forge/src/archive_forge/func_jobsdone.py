import os
import sys
import logging
import argparse
import threading
import time
from queue import Queue
import Pyro4
from gensim import utils
def jobsdone(self):
    """Wrap :attr:`~gensim.models.lsi_dispatcher.Dispatcher._jobsdone`, needed for remote access through proxies.

        Returns
        -------
        int
            Number of jobs already completed.

        """
    return self._jobsdone