import parlai.core.build_data as build_data
import os
from parlai.core.build_data import DownloadableFile
def split_punctuation(x):
    return x.replace('.', ' . ').replace('. . .', '...').replace(',', ' , ').replace(';', ' ; ').replace(':', ' : ').replace('!', ' ! ').replace('?', ' ? ').replace('"', ' " ').replace('(', ' ( ').replace(')', ' ) ')