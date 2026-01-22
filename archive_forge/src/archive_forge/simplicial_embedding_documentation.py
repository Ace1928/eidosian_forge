from dataclasses import asdict, dataclass
from typing import Optional, Type, TypeVar
import torch
from xformers import _is_triton_available

    An implementation of the "Simplicial Embeddings"_, as proposed by Lavoie et. al

    Arguments:
        - L: the number of embedding chunks
        - temperature: optional scaling parameter for the softmax operation.
            A small (<1.) temperature will lead to a sparse representation (up to one-hot),
            while a large (>1.) temperature will make the vector more uniform

    _"Simplicial Embeddings": https://arxiv.org/pdf/2204.00616.pdf
    