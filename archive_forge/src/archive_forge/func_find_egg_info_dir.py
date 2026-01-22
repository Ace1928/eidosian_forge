import os
import pkg_resources
def find_egg_info_dir(dir):
    while 1:
        try:
            filenames = os.listdir(dir)
        except OSError:
            return None
        for fn in filenames:
            if fn.endswith('.egg-info') and os.path.isdir(os.path.join(dir, fn)):
                return os.path.join(dir, fn)
        parent = os.path.dirname(dir)
        if parent == dir:
            return None
        dir = parent