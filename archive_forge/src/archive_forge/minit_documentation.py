from __future__ import annotations
from pathlib import Path
from enum import Enum
import subprocess
import shutil
import sys
import os
import re
from glob import glob
import typing as T
from mesonbuild import build, mesonlib, mlog
from mesonbuild.coredata import FORBIDDEN_TARGET_NAMES
from mesonbuild.environment import detect_ninja
from mesonbuild.templates.mesontemplates import create_meson_build
from mesonbuild.templates.samplefactory import sample_generator

    Here we generate the new Meson sample project.
    