import doctest
import re
import textwrap
import numpy as np
def output_difference(self, example, got, optionflags):
    got = [got]
    if self.text_good:
        if not self.float_size_good:
            got.append('\n\nCAUTION: tf_doctest doesn\'t work if *some* of the *float output* is hidden with a "...".')
    got.append(self.MESSAGE)
    got = '\n'.join(got)
    return super(TfDoctestOutputChecker, self).output_difference(example, got, optionflags)