from .. import __version__
import_star      = pyrex_prefix + "import_star"
import_star_set  = pyrex_prefix + "import_star_set"
def py_version_hex(major, minor=0, micro=0, release_level=0, release_serial=0):
    return major << 24 | minor << 16 | micro << 8 | release_level << 4 | release_serial