import argparse
import pathlib
import subprocess
import sys
import __main__
def test_device(device_name, shots=0, skip_ops=True, flaky_report=False, pytest_args=None, **kwargs):
    """Run the device integration tests using an installed PennyLane device.

    Args:
        device_name (str): the name of the device to test
        shots (int or None): The number of shots/samples used to estimate
            expectation values and probability. If ``shots=None``, then the
            device is run in analytic mode (where expectation values and
            probabilities are computed exactly from the quantum state).
            If not provided, the device default is used.
        skip_ops (bool): whether to skip tests that use operations not supported
            by the device
        pytest_args (list[str]): additional PyTest arguments and flags
        **kwargs: Additional device keyword args

    **Example**

    >>> from pennylane.devices.tests import test_device
    >>> test_device("default.qubit.legacy")
    ================================ test session starts =======================================
    platform linux -- Python 3.7.7, pytest-5.4.2, py-1.8.1, pluggy-0.13.1
    rootdir: /home/josh/xanadu/pennylane/pennylane/devices/tests, inifile: pytest.ini
    devices: flaky-3.6.1, cov-2.8.1, mock-3.1.0
    collected 86 items
    xanadu/pennylane/pennylane/devices/tests/test_gates.py ..............................
    ...............................                                                       [ 70%]
    xanadu/pennylane/pennylane/devices/tests/test_measurements.py .......sss...sss..sss   [ 95%]
    xanadu/pennylane/pennylane/devices/tests/test_properties.py ....                      [100%]
    ================================= 77 passed, 9 skipped in 0.78s ============================

    """
    try:
        import pytest
        import pytest_mock
        import flaky
    except ImportError as e:
        raise ImportError('The device tests requires the following Python packages:\npytest pytest_mock flaky\nThese can be installed using pip.') from e
    pytest_args = pytest_args or []
    test_dir = get_device_tests()
    cmds = ['pytest']
    cmds.append(test_dir)
    cmds.append(f'--device={device_name}')
    if shots != 0:
        cmds.append(f'--shots={shots}')
    if skip_ops:
        cmds.append('--skip-ops')
    if not flaky_report:
        cmds.append('--no-flaky-report')
    if kwargs:
        device_kwargs = ' '.join([f'{k}={v}' for k, v in kwargs.items()])
        cmds += ['--device-kwargs', device_kwargs]
    try:
        subprocess.run(cmds + pytest_args, check=not interactive)
    except subprocess.CalledProcessError as e:
        if e.returncode in range(1, 6):
            sys.exit(1)
        raise e