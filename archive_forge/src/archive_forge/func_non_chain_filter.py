import configparser
import logging
import logging.handlers
import os
import signal
import sys
from oslo_rootwrap import filters
from oslo_rootwrap import subprocess
def non_chain_filter(fltr):
    return fltr.run_as == f.run_as and (not isinstance(fltr, filters.ChainingFilter))