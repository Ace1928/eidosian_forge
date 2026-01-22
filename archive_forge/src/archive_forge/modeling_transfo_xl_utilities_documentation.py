import torch
from torch import nn

        Computes log probabilities for all \\(n\_classes\\) From:
        https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/adaptive.p

        Args:
            hidden (Tensor): a minibatch of example

        Returns:
            log-probabilities of for each class \\(c\\) in range \\(0 <= c <= n\_classes\\), where \\(n\_classes\\) is
            a parameter passed to `AdaptiveLogSoftmaxWithLoss` constructor. Shape:

            - Input: \\((N, in\_features)\\)
            - Output: \\((N, n\_classes)\\)
        