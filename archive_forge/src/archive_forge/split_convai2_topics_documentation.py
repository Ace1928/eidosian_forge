from argparse import ArgumentParser
import json
import os
import random
from parlai.projects.self_feeding.utils import extract_fb_episodes, episode_to_examples

    Converts a Fbdialog file of episodes into two self-feeding files (split by topic)

    All conversations including a word in the provided topic's bag of words will be
    separated from conversations without those words.
    