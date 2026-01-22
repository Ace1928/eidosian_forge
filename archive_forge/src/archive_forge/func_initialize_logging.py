import json
import logging
import sqlite3
from typing import List, Any, Dict, Tuple
def initialize_logging():
    logging.basicConfig(filename='evie_library_management.log', level=logging.DEBUG, format='%(asctime)s %(levelname)s:%(message)s')
    logging.info('Logging initialized for EVIE')