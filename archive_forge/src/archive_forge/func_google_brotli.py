import os
import shutil
import subprocess
import nox
@nox.session(python=['2', '3'])
def google_brotli(session):
    session.install('brotli')
    tests_impl(session, extras='socks,secure')