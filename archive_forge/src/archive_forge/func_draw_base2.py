import torch
from typing import Optional
def draw_base2(self, m: int, out: Optional[torch.Tensor]=None, dtype: torch.dtype=torch.float32) -> torch.Tensor:
    """
        Function to draw a sequence of :attr:`2**m` points from a Sobol sequence.
        Note that the samples are dependent on the previous samples. The size
        of the result is :math:`(2**m, dimension)`.

        Args:
            m (Int): The (base2) exponent of the number of points to draw.
            out (Tensor, optional): The output tensor
            dtype (:class:`torch.dtype`, optional): the desired data type of the
                                                    returned tensor.
                                                    Default: ``torch.float32``
        """
    n = 2 ** m
    total_n = self.num_generated + n
    if not total_n & total_n - 1 == 0:
        raise ValueError(f"The balance properties of Sobol' points require n to be a power of 2. {self.num_generated} points have been previously generated, then: n={self.num_generated}+2**{m}={total_n}. If you still want to do this, please use 'SobolEngine.draw()' instead.")
    return self.draw(n=n, out=out, dtype=dtype)