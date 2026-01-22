import parlai.core.build_data as build_data
import glob
import gzip
import multiprocessing
import os
import re
import sys
import time
import tqdm
import xml.etree.ElementTree as ET
from parlai.core.build_data import DownloadableFile
def conversation_to_fb_format(conversation):
    assert len(conversation) > 1
    lines = []
    for i in range(0, len(conversation), 2):
        if i + 1 < len(conversation):
            lines.append('%d %s\t%s' % (i / 2 + 1, conversation[i], conversation[i + 1]))
        else:
            lines.append('%d %s' % (i / 2 + 1, conversation[i]))
    return '\n'.join(lines)