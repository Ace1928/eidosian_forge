from ._constants import *
def getwhile(self, n, charset):
    result = ''
    for _ in range(n):
        c = self.next
        if c not in charset:
            break
        result += c
        self.__next()
    return result