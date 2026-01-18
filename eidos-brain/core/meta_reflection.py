"""Utilities for meta-level reflection and adaptation."""


class MetaReflection:
    """Provide data analysis and summarization utilities."""

    def analyze(self, data: object) -> dict:
        """Analyze ``data`` and return reflection details.

        Parameters
        ----------
        data:
            Any Python object to reflect upon.

        Returns
        -------
        dict
            Dictionary containing a representation, type name, length of the
            stringified data, and a basic summary when possible.
        """
        summary = None
        if isinstance(data, (list, tuple, set)):
            summary = f"contains {len(data)} items"
        elif isinstance(data, dict):
            summary = f"mapping with {len(data)} keys"
        elif isinstance(data, str):
            summary = f"{min(len(data), 20)} chars preview: {data[:20]}"

        return {
            "repr": repr(data),
            "type": type(data).__name__,
            "length": len(str(data)),
            "summary": summary,
        }
