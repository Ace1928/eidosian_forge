from jupyter_server.services.config.manager import ConfigManager
def get_jpserver_extensions(self, section_name=DEFAULT_SECTION_NAME):
    """Return the jpserver_extensions field from all
        config files found."""
    data = self.get(section_name)
    return data.get('ServerApp', {}).get('jpserver_extensions', {})