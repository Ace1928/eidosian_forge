import copy
from parlai.core.teachers import ParlAIDialogTeacher, FbDialogTeacher
def extract_fb_episodes(datafile):
    opt = {'datatype': 'train', 'datafile': datafile}
    episode = None
    for parley in FbDialogTeacher(opt).setup_data(datafile):
        fields, is_new_episode = parley
        if is_new_episode:
            if episode is not None:
                yield episode
            episode = []
        raw_parley = Parley(*fields)
        parley = sanitize_parley(raw_parley)
        episode.append(parley)
    yield episode