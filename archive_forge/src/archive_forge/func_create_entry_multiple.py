from parlai.core.teachers import FixedDialogTeacher
from .build import build, RESOURCES
import os
import json
def create_entry_multiple(episode):
    entries = []
    for question in episode['questions']:
        new_episode = dict()
        new_episode['asin'] = episode['asin']
        new_episode['askerID'] = question['askerID']
        new_episode['questionTime'] = question['questionTime']
        new_episode['quesitonType'] = question['questionType']
        new_episode['question'] = question['questionText']
        for answer in question['answers']:
            answer.update(new_episode)
            answer['answer'] = answer['answerText']
            entries.append([create_entry_single(answer)])
    return entries