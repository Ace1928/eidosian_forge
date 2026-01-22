Create a filtered run iterator.

        :Parameters:
            `base_iterator` : `AbstractRunIterator`
                Source of runs.
            `filter` : ``lambda object: bool``
                Function taking a value as parameter, and returning ``True``
                if the value is acceptable, and ``False`` if the default value
                should be substituted.
            `default` : object
                Default value to replace filtered values.

        