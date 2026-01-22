from typing import Union
def make_label_flat(self, var_name: str, sel: dict, isel: dict):
    """WIP."""
    var_name_str = self.var_name_to_str(var_name)
    sel_str = self.sel_to_str(sel, isel)
    if not sel_str:
        return '' if var_name_str is None else var_name_str
    if var_name_str is None:
        return sel_str
    return f'{var_name_str}[{sel_str}]'