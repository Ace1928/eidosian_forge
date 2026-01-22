import logging
import os
import sys
from pyomo.common.download import FileDownloader
def get_mcpp(downloader):
    url = 'https://github.com/omega-icl/mcpp/archive/master.zip'
    downloader.set_destination_filename(os.path.join('src', 'mcpp'))
    logger.info('Fetching MC++ from %s and installing it to %s' % (url, downloader.destination()))
    downloader.get_zip_archive(url, dirOffset=1)