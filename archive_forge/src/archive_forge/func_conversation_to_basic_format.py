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
def conversation_to_basic_format(conversation):
    assert len(conversation) > 1
    lines = []
    for i in range(len(conversation)):
        if i + 1 < len(conversation):
            lines.append('1 %s\t%s' % (conversation[i], conversation[i + 1]))
    return '\n'.join(lines)