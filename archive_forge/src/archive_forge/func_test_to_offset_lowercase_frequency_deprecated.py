import re
import pytest
from pandas._libs.tslibs import (
@pytest.mark.parametrize('freq_depr', ['2ye-mar', '2ys', '2qe', '2qs-feb', '2bqs', '2sms', '2bms', '2cbme', '2me', '2w'])
def test_to_offset_lowercase_frequency_deprecated(freq_depr):
    depr_msg = f"'{freq_depr[1:]}' is deprecated and will be removed in a "
    f"future version, please use '{freq_depr.upper()[1:]}' instead."
    with pytest.raises(FutureWarning, match=depr_msg):
        to_offset(freq_depr)