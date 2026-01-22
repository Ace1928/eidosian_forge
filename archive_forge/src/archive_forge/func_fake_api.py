from castellan.tests.unit.key_manager import mock_key_manager
def fake_api(configuration=None):
    return mock_key_manager.MockKeyManager(configuration)