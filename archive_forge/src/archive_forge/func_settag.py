import rpy2.rlike.indexing as rli
from typing import Any
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple
def settag(self, i, t):
    """
        Set tag 't' for item 'i'.

        :param i: integer (index)

        :param t: object (tag)
        """
    self.__tags[i] = t