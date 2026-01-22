from parlai.core.message import Message
from parlai.core.teachers import FixedDialogTeacher
from parlai.tasks.light_dialog.agents import DefaultTeacher as OrigLightTeacher
from parlai.tasks.light_genderation_bias.build import build
from collections import deque
from copy import deepcopy
import csv
import json
import os
import random
def get_finegrained_count(text, gender_dct):
    """
    Count the number of female, male, and neutral gendered words in a string, given the
    gender dict.
    """
    text = text.lower()
    f_count = 0
    m_count = 0
    n_count = 0
    for line in text.split('\n'):
        words = line.split(' ')
        for word in words:
            if word in gender_dct:
                if gender_dct[word]['gender'] == 'F':
                    f_count += 1
                else:
                    m_count += 1
            else:
                n_count += 1
    return (f_count, m_count, n_count)