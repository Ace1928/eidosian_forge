from typing import Union
def model_name_to_str(self, model_name):
    """WIP."""
    model_name_str = self.var_name_map.get(model_name, model_name)
    return super().model_name_to_str(model_name_str)