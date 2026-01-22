from collections import defaultdict
def display_all(self, context):
    result = []
    for target in self._targets:
        x = self.display(context, target)
        if x:
            result.append(x)
    return result