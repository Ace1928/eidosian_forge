import pytest
import os
import subprocess
import sys
import shutil
@classmethod
def get_run_env(cls):
    env = super(TestVideoWidget, cls).get_run_env()
    import kivy
    env['__KIVY_VIDEO_TEST_FNAME'] = os.path.abspath(os.path.join(kivy.kivy_examples_dir, 'widgets', 'cityCC0.mpg'))
    return env