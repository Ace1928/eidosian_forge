from argparse import ArgumentParser
import json
import os
import random
from parlai.projects.self_feeding.utils import extract_fb_episodes, episode_to_examples
def includes_topic(episode, topic):
    episode_words = ' '.join([parley.context for parley in episode]).split() + ' '.join([parley.response for parley in episode]).split()
    if TOPIC_NAME == 'family':
        return any((w in episode_words for w in topic))
    elif TOPIC_NAME == 'sports':
        episode_string = ' '.join(episode_words)
        return any((w in episode_string for w in topic))