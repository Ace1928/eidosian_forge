import parlai.core.build_data as build_data
import os
import subprocess
import shutil
import csv
import time
from parlai.core.build_data import DownloadableFile
def download_stories(path):
    documents_csv = os.path.join(path, 'documents.csv')
    tmp_dir = os.path.join(path, 'tmp')
    build_data.make_dir(tmp_dir)
    with open(documents_csv, 'r') as f:
        reader = csv.DictReader(f, delimiter=',')
        for row in reader:
            print('Downloading %s (%s)' % (row['wiki_title'], row['document_id']))
            finished = try_downloading(tmp_dir, row)
            count = 0
            while not finished and count < 5:
                if count != 0:
                    print('Retrying (%d retries left)' % (5 - count - 1))
                finished = try_downloading(tmp_dir, row)
                count += 1