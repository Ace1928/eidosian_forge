import subprocess
import sys
def test_docstring_optimization_compat():
    cmd = sys.executable + ' -OO -c "import statsmodels.api as sm"'
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out = p.communicate()
    rc = p.returncode
    assert rc == 0, out