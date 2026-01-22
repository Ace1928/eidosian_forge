from distutils.command import build_py
def test_pre_hook(cmdobj):
    print('build_ext pre-hook')