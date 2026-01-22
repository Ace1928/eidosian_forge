import pytest
from ase.build import bulk
from ase.io import write
def test_info_file_fail(cli):
    cli.ase('info', 'nonexistent_file.traj', expect_fail=True)