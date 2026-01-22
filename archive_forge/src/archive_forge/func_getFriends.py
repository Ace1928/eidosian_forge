from collections import namedtuple
def getFriends(character):
    return map(getCharacter, character.friends)