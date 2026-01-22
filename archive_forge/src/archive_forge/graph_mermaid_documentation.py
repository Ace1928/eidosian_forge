import base64
import re
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple
from langchain_core.runnables.graph import (
Renders Mermaid graph using the Mermaid.INK API.