import math
import tempfile
import warnings
from collections.abc import Sequence
from typing import List, Optional, Tuple, Union
import torch
import torchaudio
from torch import Tensor
from torchaudio._internal.module_utils import deprecated
from .filtering import highpass_biquad, treble_biquad
def frechet_distance(mu_x, sigma_x, mu_y, sigma_y):
    """Computes the Fréchet distance between two multivariate normal distributions :cite:`dowson1982frechet`.

    Concretely, for multivariate Gaussians :math:`X(\\mu_X, \\Sigma_X)`
    and :math:`Y(\\mu_Y, \\Sigma_Y)`, the function computes and returns :math:`F` as

    .. math::
        F(X, Y) = || \\mu_X - \\mu_Y ||_2^2
        + \\text{Tr}\\left( \\Sigma_X + \\Sigma_Y - 2 \\sqrt{\\Sigma_X \\Sigma_Y} \\right)

    Args:
        mu_x (torch.Tensor): mean :math:`\\mu_X` of multivariate Gaussian :math:`X`, with shape `(N,)`.
        sigma_x (torch.Tensor): covariance matrix :math:`\\Sigma_X` of :math:`X`, with shape `(N, N)`.
        mu_y (torch.Tensor): mean :math:`\\mu_Y` of multivariate Gaussian :math:`Y`, with shape `(N,)`.
        sigma_y (torch.Tensor): covariance matrix :math:`\\Sigma_Y` of :math:`Y`, with shape `(N, N)`.

    Returns:
        torch.Tensor: the Fréchet distance between :math:`X` and :math:`Y`.
    """
    if len(mu_x.size()) != 1:
        raise ValueError(f'Input mu_x must be one-dimensional; got dimension {len(mu_x.size())}.')
    if len(sigma_x.size()) != 2:
        raise ValueError(f'Input sigma_x must be two-dimensional; got dimension {len(sigma_x.size())}.')
    if sigma_x.size(0) != sigma_x.size(1) != mu_x.size(0):
        raise ValueError("Each of sigma_x's dimensions must match mu_x's size.")
    if mu_x.size() != mu_y.size():
        raise ValueError(f'Inputs mu_x and mu_y must have the same shape; got {mu_x.size()} and {mu_y.size()}.')
    if sigma_x.size() != sigma_y.size():
        raise ValueError(f'Inputs sigma_x and sigma_y must have the same shape; got {sigma_x.size()} and {sigma_y.size()}.')
    a = (mu_x - mu_y).square().sum()
    b = sigma_x.trace() + sigma_y.trace()
    c = torch.linalg.eigvals(sigma_x @ sigma_y).sqrt().real.sum()
    return a + b - 2 * c