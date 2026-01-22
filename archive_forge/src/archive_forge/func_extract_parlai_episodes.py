import copy
from parlai.core.teachers import ParlAIDialogTeacher, FbDialogTeacher
def extract_parlai_episodes(datafile):
    opt = {'datatype': 'train', 'datafile': datafile, 'parlaidialogteacher_datafile': datafile}
    episode = None
    for episode in ParlAIDialogTeacher(opt).episodes:
        episode = [Parley(**parley_dict) for parley_dict in episode]
        yield episode