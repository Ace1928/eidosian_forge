import random
import torch
def set_global_layout(self, h, layout):
    """Sets global attention layout used by the given head in the sparse attention.
        Arguments:
             h: required: an integer determining head index
             layout: required: a tensor of dimension (num_heads, num_blocks, num_blocks) containing
                sparsity layout of all head; may not be completely set at this step
        Return:
             layout: a tensor of dimension (num_heads, num_blocks, num_blocks) containing sparsity
                layout of all head in which global layout is set
        """
    num_blocks = layout.shape[1]
    if self.global_block_end_indices is None:
        for idx in self.global_block_indices:
            if idx < num_blocks:
                layout[h, idx, :] = 1
                layout[h, :, idx] = 1
    else:
        for _, (start_idx, end_idx) in enumerate(zip(self.global_block_indices, self.global_block_end_indices)):
            if start_idx < num_blocks:
                end_idx = min(end_idx, num_blocks)
                layout[h, start_idx:end_idx, :] = 1
                layout[h, :, start_idx:end_idx] = 1
    if self.attention == 'unidirectional':
        layout = torch.tril(layout)
    return layout