from parlai.core.teachers import DialogTeacher, ChunkTeacher, ChunkOutput
from parlai.core.message import Message
from .build import build
import json
import os
from typing import List, Tuple
def get_extraction_instructions(self):
    """
        If one wants to run extraction themselves on a raw wikipedia dump.
        """
    dpath = os.path.join(self.opt['datapath'], 'wikipedia', 'full')
    fname = 'enwiki-latest-pages-articles.xml.bz2'
    instructions = 'To complete the data extraction, please run the following:\nmkdir -p {download} && git clone https://github.com/attardi/wikiextractor {download}/wikiextract && cd {download}/wikiextract && python WikiExtractor.py {wikifile} --filter_disambig_pages -o {output} --json'.format(download=self.opt['download_path'], wikifile=dpath + '/' + fname, output=dpath + '/' + 'wiki_extracted')
    return instructions