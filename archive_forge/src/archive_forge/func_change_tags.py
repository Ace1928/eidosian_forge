def change_tags(self, new_tags, gone_tags):
    """Change the tags on this context.

        :param new_tags: A set of tags to add to this context.
        :param gone_tags: A set of tags to remove from this context.
        :return: The tags now current on this context.
        """
    self._tags.update(new_tags)
    self._tags.difference_update(gone_tags)
    return self.get_current_tags()