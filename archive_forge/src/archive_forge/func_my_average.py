import modin.pandas as pd
from modin.pandas.api.extensions import register_dataframe_accessor
@register_dataframe_accessor(method_name)
def my_average(self):
    return self.sum() / self.count()