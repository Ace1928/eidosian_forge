from collections import namedtuple
def getCharacter(id):
    return humanData.get(id) or droidData.get(id)