from .base import Renderer
def draw_path_collection(self, paths, path_coordinates, path_transforms, offsets, offset_coordinates, offset_order, styles, mplobj=None):
    self.output += '    draw path collection with {0} offsets\n'.format(offsets.shape[0])