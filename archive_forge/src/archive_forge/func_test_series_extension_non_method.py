import modin.pandas as pd
from modin.pandas.api.extensions import register_series_accessor
def test_series_extension_non_method():
    expected_val = 4
    attribute_name = 'four'
    register_series_accessor(attribute_name)(expected_val)
    ser = pd.Series([1, 2, 3])
    assert attribute_name in pd.series._SERIES_EXTENSIONS_.keys()
    assert pd.series._SERIES_EXTENSIONS_[attribute_name] == 4
    assert ser.four == expected_val