from kivy.config import ConfigParser
from os.path import join, dirname
def test_configparser_read():
    """Test that the ConfigParser can read a config file."""
    config = ConfigParser()
    config.read(SAMPLE_CONFIG)
    assert config.get('section', 'key') == 'value'