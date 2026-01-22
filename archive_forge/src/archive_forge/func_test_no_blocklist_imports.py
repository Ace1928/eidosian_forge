import sys
from subprocess import check_output
from textwrap import dedent
def test_no_blocklist_imports():
    check = '    import sys\n    import geoviews as gv\n\n    blocklist = {"panel", "IPython", "datashader", "iris", "dask"}\n    mods = blocklist & set(sys.modules)\n\n    if mods:\n        print(", ".join(mods), end="")\n        '
    output = check_output([sys.executable, '-c', dedent(check)])
    assert output == b''