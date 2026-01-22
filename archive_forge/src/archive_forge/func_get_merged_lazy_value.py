from jedi.inference.base_value import ValueSet, NO_VALUES
from jedi.common import monkeypatch
def get_merged_lazy_value(lazy_values):
    if len(lazy_values) > 1:
        return MergedLazyValues(lazy_values)
    else:
        return lazy_values[0]