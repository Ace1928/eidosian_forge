def set_run(self, start, end, value):
    """Set the value of a range of characters.

        :Parameters:
            `start` : int
                Start index of range.
            `end` : int
                End of range, exclusive.
            `value` : object
                Value to set over the range.

        """
    if end - start <= 0:
        return
    i = 0
    start_i = None
    start_trim = 0
    end_i = None
    end_trim = 0
    for run_i, run in enumerate(self.runs):
        count = run.count
        if i < start < i + count:
            start_i = run_i
            start_trim = start - i
        if i < end < i + count:
            end_i = run_i
            end_trim = end - i
        i += count
    if start_i is not None:
        run = self.runs[start_i]
        self.runs.insert(start_i, _Run(run.value, start_trim))
        run.count -= start_trim
        if end_i is not None:
            if end_i == start_i:
                end_trim -= start_trim
            end_i += 1
    if end_i is not None:
        run = self.runs[end_i]
        self.runs.insert(end_i, _Run(run.value, end_trim))
        run.count -= end_trim
    i = 0
    for run in self.runs:
        if start <= i and i + run.count <= end:
            run.value = value
        i += run.count
    last_run = self.runs[0]
    for run in self.runs[1:]:
        if run.value == last_run.value:
            run.count += last_run.count
            last_run.count = 0
        last_run = run
    self.runs = [r for r in self.runs if r.count > 0]