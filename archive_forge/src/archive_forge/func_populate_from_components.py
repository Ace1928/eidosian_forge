from docutils import languages, ApplicationError, TransformSpec
def populate_from_components(self, components):
    """
        Store each component's default transforms, with default priorities.
        Also, store components by type name in a mapping for later lookup.
        """
    for component in components:
        if component is None:
            continue
        self.add_transforms(component.get_transforms())
        self.components[component.component_type] = component
    self.sorted = 0
    unknown_reference_resolvers = []
    for i in components:
        unknown_reference_resolvers.extend(i.unknown_reference_resolvers)
    decorated_list = [(f.priority, f) for f in unknown_reference_resolvers]
    decorated_list.sort()
    self.unknown_reference_resolvers.extend([f[1] for f in decorated_list])