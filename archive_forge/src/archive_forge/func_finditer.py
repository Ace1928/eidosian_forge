import re
from typing import Optional, Pattern, Match, Optional
def finditer(self, *args, **kwargs):
    return self.compiled.finditer(*args, **kwargs)