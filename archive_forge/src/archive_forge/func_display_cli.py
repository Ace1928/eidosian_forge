import os
import json
import random
import tempfile
import subprocess
from parlai.core.params import ParlaiParser
from parlai.core.script import ParlaiScript, register_script
def display_cli(conversations, alt_speaker, human_speaker):
    """
    Display the conversations on the Command Line.

    :param conversations: The dialogs to be displayed
    :param alt_speaker: Name of other speaker to be used
    :param human_speaker: Name of human speaker to be used
    """
    for speaker, speech in conversations:
        if speaker == END_OF_CONVO:
            print('-' * 20 + 'END OF CONVERSATION' + '-' * 20)
        elif speaker == alt_speaker:
            print('%-15s: %s' % (speaker[:15], speech))
        else:
            prBlueBG('%-15s: %s' % (speaker[:15], speech))