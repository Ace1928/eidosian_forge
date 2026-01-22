import collections
import json
import parse_tfm
import subprocess
import sys
def find_font_path(font_name):
    try:
        font_path = subprocess.check_output(['kpsewhich', font_name])
    except OSError:
        raise RuntimeError("Couldn't find kpsewhich program, make sure you" + ' have TeX installed')
    except subprocess.CalledProcessError:
        raise RuntimeError("Couldn't find font metrics: '%s'" % font_name)
    return font_path.strip()