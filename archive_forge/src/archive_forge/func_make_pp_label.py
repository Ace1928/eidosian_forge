from typing import Union
def make_pp_label(self, var_name, pp_var_name, sel, isel):
    """WIP."""
    names = self.var_pp_to_str(var_name, pp_var_name)
    return self.make_label_vert(names, sel, isel)