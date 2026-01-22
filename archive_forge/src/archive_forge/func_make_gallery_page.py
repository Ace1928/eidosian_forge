import os
import re
from os.path import sep
from os.path import join as slash  # just like that name better
from os.path import dirname, abspath
import kivy
from kivy.logger import Logger
import textwrap
def make_gallery_page(infos):
    """ return string of the rst (Restructured Text) of the gallery page,
    showing information on all screenshots found.
    """
    gallery_top = "\nGallery\n-------\n\n.. _Tutorials:  ../tutorials-index.html\n\n.. container:: title\n\n    This gallery lets you explore the many examples included with Kivy.\n    Click on any screenshot to see the code.\n\nThis gallery contains:\n\n    * Examples from the examples/ directory that show specific capabilities of\n      different libraries and features of Kivy.\n    * Demonstrations from the examples/demos/ directory that explore many of\n      Kivy's abilities.\n\nThere are more Kivy programs elsewhere:\n\n    * Tutorials_ walks through the development of complete Kivy applications.\n    * Unit tests found in the source code under the subdirectory kivy/tests/\n      can also be useful.\n\nWe hope your journey into learning Kivy is exciting and fun!\n\n"
    output = [gallery_top]
    for info in infos:
        output.append('\n**{title}** (:doc:`{source}<gen__{dunder}>`)\n\n{description}\n.. image:: ../images/examples/{dunder}.png\n  :width:  216pt\n  :align:  left\n  :target: gen__{dunder}.html'.format(**info))
    return '\n'.join(output) + '\n'