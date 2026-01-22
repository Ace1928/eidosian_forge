from io import StringIO
def isRootMatch(self, elem):
    if (self.elementName == None or self.elementName == elem.name) and self.matchesPredicates(elem):
        if self.childLocation != None:
            for c in elem.elements():
                if self.childLocation.matches(c):
                    return True
        else:
            return True
    return False