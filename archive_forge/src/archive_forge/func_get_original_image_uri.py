from sphinx.environment import BuildEnvironment
def get_original_image_uri(self, name: str) -> str:
    """Get the original image URI."""
    while name in self.env.original_image_uri:
        name = self.env.original_image_uri[name]
    return name