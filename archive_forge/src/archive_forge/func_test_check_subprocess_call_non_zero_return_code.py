import sys
import re
from joblib.testing import raises, check_subprocess_call
def test_check_subprocess_call_non_zero_return_code():
    code_with_non_zero_exit = '\n'.join(['import sys', 'print("writing on stdout")', 'sys.stderr.write("writing on stderr")', 'sys.exit(123)'])
    pattern = re.compile('Non-zero return code: 123.+Stdout:\nwriting on stdout.+Stderr:\nwriting on stderr', re.DOTALL)
    with raises(ValueError) as excinfo:
        check_subprocess_call([sys.executable, '-c', code_with_non_zero_exit])
    excinfo.match(pattern)