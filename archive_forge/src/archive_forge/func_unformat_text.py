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
def unformat_text(text):
    for punc in PUNCTUATION_LST:
        text = text.replace(punc[0], punc[1])
    return text