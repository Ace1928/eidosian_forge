from __future__ import annotations
import argparse, datetime, glob, json, os, platform, shutil, sys, tempfile, time
import cProfile as profile
from pathlib import Path
import typing as T
from . import build, coredata, environment, interpreter, mesonlib, mintro, mlog
from .mesonlib import MesonException
def finalize_postconf_hooks(self, b: build.Build, intr: interpreter.Interpreter) -> None:
    b.devenv.append(intr.backend.get_devenv())
    for mod in intr.modules.values():
        mod.postconf_hook(b)