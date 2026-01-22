import linecache
import re

    Set the trace filter mode.

    mode: Whether to enable the trace hook.
      True: Trace filtering on (skipping methods tagged @DontTrace)
      False: Trace filtering off (trace methods tagged @DontTrace)
      None/default: Toggle trace filtering.
    