import warnings
from io import StringIO
def body__get(self):
    if self.str is None:
        return self.output.getvalue()
    else:
        return self.str