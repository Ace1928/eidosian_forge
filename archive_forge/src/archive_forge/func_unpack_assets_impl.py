import os
import sys
import json
from os.path import isdir
import subprocess
import pathlib
import setuptools
from setuptools import Command
from setuptools.command.develop import develop
from setuptools.command.install import install
from setuptools.command.install_lib import install_lib
from distutils.command.build import build
from setuptools.dist import Distribution
import shutil
def unpack_assets_impl(index, asset_dir, output_dir):
    for k, v in index['objects'].items():
        asset_hash = v['hash']
        src = os.path.join(asset_dir, 'objects', asset_hash[:2], asset_hash)
        dst = os.path.join(output_dir, k)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(src, dst)