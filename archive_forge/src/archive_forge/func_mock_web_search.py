import json
import logging
import sqlite3
from typing import List, Any, Dict, Tuple
def mock_web_search(query: str) -> List[str]:
    return ['Mocked related topic 1', 'Mocked related topic 2']