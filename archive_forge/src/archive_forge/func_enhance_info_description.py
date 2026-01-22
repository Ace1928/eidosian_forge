import os
import re
from os.path import sep
from os.path import join as slash  # just like that name better
from os.path import dirname, abspath
import kivy
from kivy.logger import Logger
import textwrap
def enhance_info_description(info, line_length=79):
    """ Using the info['description'], add fields to info.

    info['files'] is the source filename and any filenames referenced by the
    magic words in the description, e.g. 'the file xxx.py' or
    'The image this.png'. These are as written in the description, do
    not allow ../dir notation, and are relative to the source directory.

    info['enhanced_description'] is the description, as an array of
    paragraphs where each paragraph is an array of lines wrapped to width
    line_length. This enhanced description include the rst links to
    the files of info['files'].
    """
    paragraphs = info['description'].split('\n\n')
    lines = [paragraph.replace('\n', '$newline$') for paragraph in paragraphs]
    text = '\n'.join(lines)
    info['files'] = [info['file'] + '.' + info['ext']]
    regex = '[tT]he (?:file|image) ([\\w\\/]+\\.\\w+)'
    for name in re.findall(regex, text):
        if name not in info['files']:
            info['files'].append(name)
    folder = '_'.join(info['source'].split(sep)[:-1]) + '_'
    text = re.sub('([tT]he (?:file|image) )([\\w\\/]+\\.\\w+)', '\\1:ref:`\\2 <$folder$\\2>`', text)
    text = text.replace('$folder$', folder)
    lines = [line.replace('$newline$', '\n') for line in text.split('\n')]
    paragraphs = [textwrap.wrap(line, line_length) if not line.startswith(' ') else [line] for line in lines]
    info['enhanced_description'] = paragraphs