import os, subprocess, json
@classmethod
def extract_directory_tag(cls, setup_path, reponame):
    setup_dir = os.path.split(setup_path)[-1]
    prefix = reponame + '-'
    if setup_dir.startswith(prefix):
        tag = setup_dir[len(prefix):]
        if tag not in ['', 'master', 'main'] and '.' in tag:
            return tag
    return None