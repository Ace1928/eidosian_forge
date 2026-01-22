import json
import os
import os.path
import tempfile
from typing import List, Optional
from gym import error, logger
Closes the environment correctly when the recorder is deleted.