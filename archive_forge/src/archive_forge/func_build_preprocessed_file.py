import functools
import os
import re
import tempfile
from nltk.corpus.reader.util import concat
from nltk.corpus.reader.xmldocs import XMLCorpusReader, XMLCorpusView
def build_preprocessed_file(self):
    try:
        fr = open(self.read_file)
        fw = self.write_file
        line = ' '
        while len(line):
            line = fr.readline()
            x = re.split('nkjp:[^ ]* ', line)
            ret = ' '.join(x)
            x = re.split('<nkjp:paren>', ret)
            ret = ' '.join(x)
            x = re.split('</nkjp:paren>', ret)
            ret = ' '.join(x)
            x = re.split('<choice>', ret)
            ret = ' '.join(x)
            x = re.split('</choice>', ret)
            ret = ' '.join(x)
            fw.write(ret)
        fr.close()
        fw.close()
        return self.write_file.name
    except Exception as e:
        self.remove_preprocessed_file()
        raise Exception from e