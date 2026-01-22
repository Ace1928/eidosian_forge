import random
def sentence():
    """
    Return a randomly generated sentence of lorem ipsum text.

    The first word is capitalized, and the sentence ends in either a period or
    question mark. Commas are added at random.
    """
    sections = [' '.join(random.sample(WORDS, random.randint(3, 12))) for i in range(random.randint(1, 5))]
    s = ', '.join(sections)
    return '%s%s%s' % (s[0].upper(), s[1:], random.choice('?.'))