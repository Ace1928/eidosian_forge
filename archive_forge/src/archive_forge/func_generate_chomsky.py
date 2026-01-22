from actual books and articles written by Noam Chomsky.
import random
import textwrap
from itertools import chain, islice
def generate_chomsky(times=5, line_length=72):
    parts = []
    for part in (leadins, subjects, verbs, objects):
        phraselist = list(map(str.strip, part.splitlines()))
        random.shuffle(phraselist)
        parts.append(phraselist)
    output = chain.from_iterable(islice(zip(*parts), 0, times))
    print(textwrap.fill(' '.join(output), line_length))