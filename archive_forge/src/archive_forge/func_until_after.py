def until_after(self, pattern):
    result = self._create()
    result._until_after = True
    result._until_pattern = self._input.get_regexp(pattern)
    result._update()
    return result