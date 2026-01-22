import re
import itertools
@classmethod
def maskScoreRule4(cls, modules):
    cellCount = len(modules) ** 2
    count = sum((sum(row) for row in modules))
    return 10 * (abs(100 * count // cellCount - 50) // 5)