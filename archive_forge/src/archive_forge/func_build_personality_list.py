import os
from parlai.core import build_data
from parlai.core.opt import Opt
def build_personality_list(opt: Opt):
    dpath = os.path.join(opt['datapath'], TASK_FOLDER_NAME)
    version = 'v1.0'
    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + ']')
        if build_data.built(dpath):
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)
        for downloadable_file in PERSONALITY_LIST_RESOURCES:
            downloadable_file.download_file(dpath)
        build_data.mark_done(dpath, version_string=version)