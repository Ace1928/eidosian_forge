import pytest
import IPython.testing.tools as tt
def test_profile_list_help():
    tt.help_all_output_test('profile list')