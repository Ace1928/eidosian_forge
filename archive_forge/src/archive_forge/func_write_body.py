import time
from ase.utils import writer
from ase.io.utils import PlottingVariables, make_patch_list
def write_body(self, fd, renderer):
    patch_list = make_patch_list(self)
    for patch in patch_list:
        patch.draw(renderer)