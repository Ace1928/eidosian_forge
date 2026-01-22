import numpy as np
def color_gradient(size, p1, p2=None, vector=None, r=None, col1=0, col2=1.0, shape='linear', offset=0):
    """Draw a linear, bilinear, or radial gradient.
    
    The result is a picture of size ``size``, whose color varies
    gradually from color `col1` in position ``p1`` to color ``col2``
    in position ``p2``.
    
    If it is a RGB picture the result must be transformed into
    a 'uint8' array to be displayed normally:
     
     
    Parameters
    ------------      
    
    size
        Size (width, height) in pixels of the final picture/array.
    
    p1, p2
        Coordinates (x,y) in pixels of the limit point for ``col1``
        and ``col2``. The color 'before' ``p1`` is ``col1`` and it
        gradually changes in the direction of ``p2`` until it is ``col2``
        when it reaches ``p2``.
    
    vector
        A vector [x,y] in pixels that can be provided instead of ``p2``.
        ``p2`` is then defined as (p1 + vector).
    
    col1, col2
        Either floats between 0 and 1 (for gradients used in masks)
        or [R,G,B] arrays (for colored gradients).
                         
    shape
        'linear', 'bilinear', or 'circular'.
        In a linear gradient the color varies in one direction,
        from point ``p1`` to point ``p2``.
        In a bilinear gradient it also varies symetrically form ``p1``
        in the other direction.
        In a circular gradient it goes from ``col1`` to ``col2`` in all
        directions.
    
    offset
        Real number between 0 and 1 indicating the fraction of the vector
        at which the gradient actually starts. For instance if ``offset``
        is 0.9 in a gradient going from p1 to p2, then the gradient will
        only occur near p2 (before that everything is of color ``col1``)
        If the offset is 0.9 in a radial gradient, the gradient will
        occur in the region located between 90% and 100% of the radius,
        this creates a blurry disc of radius d(p1,p2).  
    
    Returns
    --------
    
    image
        An Numpy array of dimensions (W,H,ncolors) of type float
        representing the image of the gradient.
        
    
    Examples
    ---------
    
    >>> grad = color_gradient(blabla).astype('uint8')
    
    """
    w, h = size
    col1 = np.array(col1).astype(float)
    col2 = np.array(col2).astype(float)
    if shape == 'bilinear':
        if vector is None:
            vector = np.array(p2) - np.array(p1)
        m1, m2 = [color_gradient(size, p1, vector=v, col1=1.0, col2=0, shape='linear', offset=offset) for v in [vector, -vector]]
        arr = np.maximum(m1, m2)
        if col1.size > 1:
            arr = np.dstack(3 * [arr])
        return arr * col1 + (1 - arr) * col2
    p1 = np.array(p1[::-1]).astype(float)
    if vector is None and p2:
        p2 = np.array(p2[::-1])
        vector = p2 - p1
    else:
        vector = np.array(vector[::-1])
        p2 = p1 + vector
    if vector:
        norm = np.linalg.norm(vector)
    M = np.dstack(np.meshgrid(range(w), range(h))[::-1]).astype(float)
    if shape == 'linear':
        n_vec = vector / norm ** 2
        p1 = p1 + offset * vector
        arr = (M - p1).dot(n_vec) / (1 - offset)
        arr = np.minimum(1, np.maximum(0, arr))
        if col1.size > 1:
            arr = np.dstack(3 * [arr])
        return arr * col1 + (1 - arr) * col2
    elif shape == 'radial':
        if r is None:
            r = norm
        if r == 0:
            arr = np.ones((h, w))
        else:
            arr = np.sqrt(((M - p1) ** 2).sum(axis=2)) - offset * r
            arr = arr / ((1 - offset) * r)
            arr = np.minimum(1.0, np.maximum(0, arr))
        if col1.size > 1:
            arr = np.dstack(3 * [arr])
        return (1 - arr) * col1 + arr * col2