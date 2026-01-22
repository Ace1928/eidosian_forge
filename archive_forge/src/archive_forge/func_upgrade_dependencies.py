import logging
import os
import shutil
import subprocess
import sys
import sysconfig
import types
def upgrade_dependencies(self, context):
    logger.debug(f'Upgrading {CORE_VENV_DEPS} packages in {context.bin_path}')
    self._call_new_python(context, '-m', 'pip', 'install', '--upgrade', *CORE_VENV_DEPS)