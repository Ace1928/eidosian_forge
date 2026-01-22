import random
import torch
def set_local_layout(self, h, layout):
    """Sets local attention layout used by the given head in the sparse attention.
        Arguments:
             h: required: an integer determining head index
             layout: required: a tensor of dimension (num_heads, num_blocks, num_blocks) containing
                sparsity layout of all head; may not be completely set at this step
        Return:
             layout: a tensor of dimension (num_heads, num_blocks, num_blocks) containing sparsity
                layout of all head in which local layout is set
        """
    num_blocks = layout.shape[1]
    start_block_idx = 0
    end_block_idx = 0
    for block_size in self.local_window_blocks:
        end_block_idx += block_size
        end_block_idx = min(end_block_idx, num_blocks)
        for row in range(start_block_idx, end_block_idx):
            for col in range(start_block_idx, row + 1 if self.attention == 'unidirectional' else end_block_idx):
                layout[h, row, col] = 1
        start_block_idx += block_size
    for i in range(start_block_idx, num_blocks, block_size):
        end_block_idx = min(i + block_size, num_blocks)
        for row in range(i, end_block_idx):
            for col in range(i, row + 1 if self.attention == 'unidirectional' else end_block_idx):
                layout[h, row, col] = 1
    return layout