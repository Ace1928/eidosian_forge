from __future__ import absolute_import
import sys
def show_as_blocks(self, block_size=100):
    """
        Show colors in the IPython Notebook using ipythonblocks.

        Parameters
        ----------
        block_size : int, optional
            Size of displayed blocks.

        """
    from ipythonblocks import BlockGrid
    grid = BlockGrid(self.number, 1, block_size=block_size)
    for block, color in zip(grid, self.colors):
        block.rgb = color
    grid.show()