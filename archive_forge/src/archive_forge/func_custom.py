import sys
def custom(self, moduleName, funcName):
    """Goes and gets the Python object and adds it to the story"""
    self.endPara()
    self._results.append(('Custom', moduleName, funcName))