import random
def random_name() -> str:
    """Generate a random name."""
    adjective = random.choice(adjectives)
    noun = random.choice(nouns)
    number = random.randint(1, 100)
    return f'{adjective}-{noun}-{number}'