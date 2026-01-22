import numpy as np
from matplotlib import _api
from matplotlib.path import Path
def set_vertices_and_codes(self, vertices, codes):
    offset = 1.0 / self.num_rows
    shape_vertices = self.shape_vertices * offset * self.size
    shape_codes = self.shape_codes
    if not self.filled:
        shape_vertices = np.concatenate([shape_vertices, shape_vertices[::-1] * 0.9])
        shape_codes = np.concatenate([shape_codes, shape_codes])
    vertices_parts = []
    codes_parts = []
    for row in range(self.num_rows + 1):
        if row % 2 == 0:
            cols = np.linspace(0, 1, self.num_rows + 1)
        else:
            cols = np.linspace(offset / 2, 1 - offset / 2, self.num_rows)
        row_pos = row * offset
        for col_pos in cols:
            vertices_parts.append(shape_vertices + [col_pos, row_pos])
            codes_parts.append(shape_codes)
    np.concatenate(vertices_parts, out=vertices)
    np.concatenate(codes_parts, out=codes)