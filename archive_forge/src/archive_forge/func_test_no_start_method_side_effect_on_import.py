import sys
import joblib
from joblib.testing import check_subprocess_call
from joblib.test.common import with_multiprocessing
@with_multiprocessing
def test_no_start_method_side_effect_on_import():
    code = 'if True:\n        import joblib\n        import multiprocessing as mp\n        # The following line would raise RuntimeError if the\n        # start_method is already set.\n        mp.set_start_method("loky")\n    '
    check_subprocess_call([sys.executable, '-c', code])