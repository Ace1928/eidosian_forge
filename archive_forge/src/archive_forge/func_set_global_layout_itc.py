import random
import torch
def set_global_layout_itc(self, h, layout):
    """Sets global attention layout used by the given head in the sparse attention.
        Arguments:
             h: required: an integer determining head index
             layout: required: a tensor of dimension (num_heads, num_blocks, num_blocks) containing
                sparsity layout of all head; may not be completely set at this step
        Return:
             layout: a tensor of dimension (num_heads, num_blocks, num_blocks) containing sparsity layout
                of all head in which global layout is set
        """
    num_blocks = layout.shape[1]
    if num_blocks < self.num_global_blocks:
        raise ValueError(f'Number of global blocks, {self.num_global_blocks}, must be smaller than overall number\n                of blocks in a row, {num_blocks}!')
    layout[h, 0:self.num_global_blocks, :] = 1
    layout[h, :, 0:self.num_global_blocks] = 1
    if self.attention == 'unidirectional':
        layout = torch.tril(layout)
    return layout