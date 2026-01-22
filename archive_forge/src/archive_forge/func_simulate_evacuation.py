import logging
import os
import pathlib
import sys
import time
import pytest
def simulate_evacuation():
    logging.getLogger().addHandler(logging.StreamHandler())
    handler = NuclearReactorMonitoringHandler()
    logging.getLogger().addHandler(handler)
    nuclear_core_logger = logging.getLogger('powerstation.core')
    nuclear_core_logger.info('Core temperature nominal')
    assert handler.NUCLEAR_REACTOR_STATUS == 'Nominal'
    nuclear_core_logger.critical('Radioactive gas leak')
    assert handler.NUCLEAR_REACTOR_STATUS == 'Evacuated'