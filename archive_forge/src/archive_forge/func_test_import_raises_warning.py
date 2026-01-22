import textwrap
import pytest
from sklearn.utils import _IS_WASM
from sklearn.utils._testing import assert_run_python_script_without_output
@pytest.mark.xfail(_IS_WASM, reason='cannot start subprocess')
def test_import_raises_warning():
    code = '\n    import pytest\n    with pytest.warns(UserWarning, match="it is not needed to import"):\n        from sklearn.experimental import enable_hist_gradient_boosting  # noqa\n    '
    pattern = 'it is not needed to import enable_hist_gradient_boosting anymore'
    assert_run_python_script_without_output(textwrap.dedent(code), pattern=pattern)