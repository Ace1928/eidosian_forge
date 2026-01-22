import os
import re
import shutil
import sys
def get_exec(self, exec_dirs=None):
    """Returns existing executable, or empty string if none found."""
    exec_dirs = exec_dirs or []
    if self.real_exec is not None:
        return self.real_exec
    if os.path.isabs(self.exec_path):
        if os.access(self.exec_path, os.X_OK):
            self.real_exec = self.exec_path
    else:
        for binary_path in exec_dirs:
            expanded_path = os.path.join(binary_path, self.exec_path)
            if os.access(expanded_path, os.X_OK):
                self.real_exec = expanded_path
                break
    return self.real_exec