import logging
import re
def replace_tildes(comp, loose):
    return ' '.join([replace_tilde(c, loose) for c in re.split('\\s+', comp.strip())])