import gast as ast
import os
import re
from time import time

        High-level function to call a `transformation' on a `node'.
        If the transformation is an analysis, the result of the analysis
        is displayed.
        