import warnings
def headeritems(self):
    result = []
    for key, value in self.items():
        if isinstance(value, list):
            for v in value:
                result.append((key, str(v)))
        else:
            result.append((key, str(value)))
    return result