from distutils.command import build_py
def test_post_hook(cmdobj):
    print('build_ext post-hook')