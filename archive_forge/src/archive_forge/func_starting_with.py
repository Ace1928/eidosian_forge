def starting_with(self, pattern):
    result = self._create()
    result._starting_pattern = self._input.get_regexp(pattern)
    result._update()
    return result