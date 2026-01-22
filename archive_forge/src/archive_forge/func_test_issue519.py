import subprocess
import sys
import pytest
@pytest.mark.slow
def test_issue519():
    """
    Test ability of Thinc mypy plugin to handle variadic arguments.

    This test can take up to 45 seconds, and is thus marked as slow.
    """
    parent_module_name = __name__[:__name__.rfind('.')]
    program_text = importlib_resources.read_text(parent_module_name, 'program.py')
    subprocess.run([sys.executable, '-m', 'mypy', '--command', program_text], check=True)