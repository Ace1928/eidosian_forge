import numpy as np
def color_split(size, x=None, y=None, p1=None, p2=None, vector=None, col1=0, col2=1.0, grad_width=0):
    """Make an image splitted in 2 colored regions.
    
    Returns an array of size ``size`` divided in two regions called 1 and
    2 in wht follows, and which will have colors col& and col2
    respectively.
    
    Parameters
    -----------
    
    x: (int)
        If provided, the image is splitted horizontally in x, the left
        region being region 1.
            
    y: (int)
        If provided, the image is splitted vertically in y, the top region
        being region 1.
    
    p1,p2:
        Positions (x1,y1),(x2,y2) in pixels, where the numbers can be
        floats. Region 1 is defined as the whole region on the left when
        going from ``p1`` to ``p2``.
    
    p1, vector:
        ``p1`` is (x1,y1) and vector (v1,v2), where the numbers can be
        floats. Region 1 is then the region on the left when starting
        in position ``p1`` and going in the direction given by ``vector``.
         
    gradient_width
        If not zero, the split is not sharp, but gradual over a region of
        width ``gradient_width`` (in pixels). This is preferable in many
        situations (for instance for antialiasing). 
     
    
    Examples
    ---------
    
    >>> size = [200,200]
    >>> # an image with all pixels with x<50 =0, the others =1
    >>> color_split(size, x=50, col1=0, col2=1)
    >>> # an image with all pixels with y<50 red, the others green
    >>> color_split(size, x=50, col1=[255,0,0], col2=[0,255,0])
    >>> # An image splitted along an arbitrary line (see below) 
    >>> color_split(size, p1=[20,50], p2=[25,70] col1=0, col2=1)
            
    """
    if grad_width or (x is None and y is None):
        if p2 is not None:
            vector = np.array(p2) - np.array(p1)
        elif x is not None:
            vector = np.array([0, -1.0])
            p1 = np.array([x, 0])
        elif y is not None:
            vector = np.array([1.0, 0.0])
            p1 = np.array([0, y])
        x, y = vector
        vector = np.array([y, -x]).astype('float')
        norm = np.linalg.norm(vector)
        vector = max(0.1, grad_width) * vector / norm
        return color_gradient(size, p1, vector=vector, col1=col1, col2=col2, shape='linear')
    else:
        w, h = size
        shape = (h, w) if np.isscalar(col1) else (h, w, len(col1))
        arr = np.zeros(shape)
        if x:
            arr[:, :x] = col1
            arr[:, x:] = col2
        elif y:
            arr[:y] = col1
            arr[y:] = col2
        return arr
    print('Arguments in color_split not understood !')
    raise