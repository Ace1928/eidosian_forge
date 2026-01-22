from __future__ import annotations
from enum import Enum
from typing import Optional, List, Dict, Any, Union, Literal, TYPE_CHECKING
@property
def vector_op(self) -> str:
    """
        Returns the Vector Op Name

        vector_l2_ops = L2 distance
        vector_ip_ops = Inner Product
        vector_cosine_ops = Cosine Distance
        """
    if self == VectorDistance.EUCLIDEAN:
        return 'vector_l2_ops'
    elif self == VectorDistance.COSINE:
        return 'vector_cosine_ops'
    return 'vector_ip_ops'