import os
import shlex
import sys
from pbr import find_package
from pbr.hooks import base
def get_man_sections(self):
    man_sections = dict()
    manpages = self.pbr_config['manpages']
    for manpage in manpages.split():
        section_number = manpage.strip()[-1]
        section = man_sections.get(section_number, list())
        section.append(manpage.strip())
        man_sections[section_number] = section
    return man_sections