import unittest
from ..pass_manager import (
Make sure we can construct the PassManager twice and not share any
        state between them