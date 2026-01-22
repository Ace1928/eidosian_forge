def stem_sentence(self, txt):
    """Stem the sentence `txt`.

        Parameters
        ----------
        txt : str
            Input sentence.

        Returns
        -------
        str
            Stemmed sentence.

        Examples
        --------
        .. sourcecode:: pycon

            >>> from gensim.parsing.porter import PorterStemmer
            >>> p = PorterStemmer()
            >>> p.stem_sentence("Wow very nice woman with apple")
            'wow veri nice woman with appl'

        """
    return ' '.join((self.stem(x) for x in txt.split()))