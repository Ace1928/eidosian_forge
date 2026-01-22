import array
import copy
import copyreg
import warnings
def init_type(self, *args, **kargs):
    """Replace the __init__ function of the new type, in order to
            add attributes that were defined with **kargs to the instance.
            """
    for obj_name, obj in dict_inst.items():
        setattr(self, obj_name, obj())
    if base.__init__ is not object.__init__:
        base.__init__(self, *args, **kargs)