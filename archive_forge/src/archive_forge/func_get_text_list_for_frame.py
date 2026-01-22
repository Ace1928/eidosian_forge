import time
from _pydev_bundle._pydev_filesystem_encoding import getfilesystemencoding
from _pydev_bundle._pydev_saved_modules import threading
from _pydevd_bundle import pydevd_xml
from _pydevd_bundle.pydevd_constants import GlobalDebuggerHolder
from _pydevd_bundle.pydevd_constants import get_thread_id
from _pydevd_bundle.pydevd_net_command import NetCommand
from _pydevd_bundle.pydevd_concurrency_analyser.pydevd_thread_wrappers import ObjectWrapper, wrap_attr
import pydevd_file_utils
from _pydev_bundle import pydev_log
import sys
from urllib.parse import quote
def get_text_list_for_frame(frame):
    curFrame = frame
    cmdTextList = []
    try:
        while curFrame:
            myId = str(id(curFrame))
            if curFrame.f_code is None:
                break
            myName = curFrame.f_code.co_name
            if myName is None:
                break
            absolute_filename = pydevd_file_utils.get_abs_path_real_path_and_base_from_frame(curFrame)[0]
            my_file, _applied_mapping = pydevd_file_utils.map_file_to_client(absolute_filename)
            myLine = str(curFrame.f_lineno)
            variables = ''
            cmdTextList.append('<frame id="%s" name="%s" ' % (myId, pydevd_xml.make_valid_xml_value(myName)))
            cmdTextList.append('file="%s" line="%s">' % (quote(my_file, '/>_= \t'), myLine))
            cmdTextList.append(variables)
            cmdTextList.append('</frame>')
            curFrame = curFrame.f_back
    except:
        pydev_log.exception()
    return cmdTextList