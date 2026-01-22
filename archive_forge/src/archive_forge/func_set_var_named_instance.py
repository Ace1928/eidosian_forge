import io
import sys
from ctypes import *
import ctypes.util
import struct
from freetype.raw import *
def set_var_named_instance(self, instance_name):
    """
        Set instance by name. This will work with any FreeType with variable support
        (for our purposes: v2.8.1 or later). If the actual FT_Set_Named_Instance()
        function is available (v2.9.1 or later), we use it (which, despite what you might
        expect from its name, sets instances by *index*). Otherwise we just use the coords
        of the named instance (if found) and call self.set_var_design_coords.
        """
    have_func = freetype.version() >= (2, 9, 1)
    vsi = self.get_variation_info()
    for inst_idx, inst in enumerate(vsi.instances, start=1):
        if inst.name == instance_name:
            if have_func:
                error = FT_Set_Named_Instance(self._FT_Face, inst_idx)
            else:
                error = self.set_var_design_coords(inst.coords)
            if error:
                raise FT_Exception(error)
            break