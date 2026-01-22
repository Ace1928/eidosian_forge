from __future__ import unicode_literals
from .nodes import FilterNode, filter_operator
from ._utils import escape_chars
@filter_operator()
def drawtext(stream, text=None, x=0, y=0, escape_text=True, **kwargs):
    """Draw a text string or text from a specified file on top of a video, using the libfreetype library.

    To enable compilation of this filter, you need to configure FFmpeg with ``--enable-libfreetype``. To enable default
    font fallback and the font option you need to configure FFmpeg with ``--enable-libfontconfig``. To enable the
    text_shaping option, you need to configure FFmpeg with ``--enable-libfribidi``.

    Args:
        box: Used to draw a box around text using the background color. The value must be either 1 (enable) or 0
            (disable). The default value of box is 0.
        boxborderw: Set the width of the border to be drawn around the box using boxcolor. The default value of
            boxborderw is 0.
        boxcolor: The color to be used for drawing box around text. For the syntax of this option, check the "Color"
            section in the ffmpeg-utils manual.  The default value of boxcolor is "white".
        line_spacing: Set the line spacing in pixels of the border to be drawn around the box using box. The default
            value of line_spacing is 0.
        borderw: Set the width of the border to be drawn around the text using bordercolor. The default value of
            borderw is 0.
        bordercolor: Set the color to be used for drawing border around text. For the syntax of this option, check the
            "Color" section in the ffmpeg-utils manual.  The default value of bordercolor is "black".
        expansion: Select how the text is expanded. Can be either none, strftime (deprecated) or normal (default). See
            the Text expansion section below for details.
        basetime: Set a start time for the count. Value is in microseconds. Only applied in the deprecated strftime
            expansion mode. To emulate in normal expansion mode use the pts function, supplying the start time (in
            seconds) as the second argument.
        fix_bounds: If true, check and fix text coords to avoid clipping.
        fontcolor: The color to be used for drawing fonts. For the syntax of this option, check the "Color" section in
            the ffmpeg-utils manual.  The default value of fontcolor is "black".
        fontcolor_expr: String which is expanded the same way as text to obtain dynamic fontcolor value. By default
            this option has empty value and is not processed. When this option is set, it overrides fontcolor option.
        font: The font family to be used for drawing text. By default Sans.
        fontfile: The font file to be used for drawing text. The path must be included. This parameter is mandatory if
            the fontconfig support is disabled.
        alpha: Draw the text applying alpha blending. The value can be a number between 0.0 and 1.0. The expression
            accepts the same variables x, y as well. The default value is 1. Please see fontcolor_expr.
        fontsize: The font size to be used for drawing text. The default value of fontsize is 16.
        text_shaping: If set to 1, attempt to shape the text (for example, reverse the order of right-to-left text and
            join Arabic characters) before drawing it. Otherwise, just draw the text exactly as given. By default 1 (if
            supported).
        ft_load_flags: The flags to be used for loading the fonts. The flags map the corresponding flags supported by
            libfreetype, and are a combination of the following values:

            * ``default``
            * ``no_scale``
            * ``no_hinting``
            * ``render``
            * ``no_bitmap``
            * ``vertical_layout``
            * ``force_autohint``
            * ``crop_bitmap``
            * ``pedantic``
            * ``ignore_global_advance_width``
            * ``no_recurse``
            * ``ignore_transform``
            * ``monochrome``
            * ``linear_design``
            * ``no_autohint``

            Default value is "default".  For more information consult the documentation for the FT_LOAD_* libfreetype
            flags.
        shadowcolor: The color to be used for drawing a shadow behind the drawn text. For the syntax of this option,
            check the "Color" section in the ffmpeg-utils manual.  The default value of shadowcolor is "black".
        shadowx: The x offset for the text shadow position with respect to the position of the text. It can be either
            positive or negative values. The default value is "0".
        shadowy: The y offset for the text shadow position with respect to the position of the text. It can be either
            positive or negative values. The default value is "0".
        start_number: The starting frame number for the n/frame_num variable. The default value is "0".
        tabsize: The size in number of spaces to use for rendering the tab. Default value is 4.
        timecode: Set the initial timecode representation in "hh:mm:ss[:;.]ff" format. It can be used with or without
            text parameter. timecode_rate option must be specified.
        rate: Set the timecode frame rate (timecode only).
        timecode_rate: Alias for ``rate``.
        r: Alias for ``rate``.
        tc24hmax: If set to 1, the output of the timecode option will wrap around at 24 hours. Default is 0 (disabled).
        text: The text string to be drawn. The text must be a sequence of UTF-8 encoded characters. This parameter is
            mandatory if no file is specified with the parameter textfile.
        textfile: A text file containing text to be drawn. The text must be a sequence of UTF-8 encoded characters.
            This parameter is mandatory if no text string is specified with the parameter text.  If both text and
            textfile are specified, an error is thrown.
        reload: If set to 1, the textfile will be reloaded before each frame. Be sure to update it atomically, or it
            may be read partially, or even fail.
        x: The expression which specifies the offset where text will be drawn within the video frame. It is relative to
            the left border of the output image. The default value is "0".
        y: The expression which specifies the offset where text will be drawn within the video frame. It is relative to
            the top border of the output image. The default value is "0".  See below for the list of accepted constants
            and functions.

    Expression constants:
        The parameters for x and y are expressions containing the following constants and functions:
         - dar: input display aspect ratio, it is the same as ``(w / h) * sar``
         - hsub: horizontal chroma subsample values. For example for the pixel format "yuv422p" hsub is 2 and vsub
           is 1.
         - vsub: vertical chroma subsample values. For example for the pixel format "yuv422p" hsub is 2 and vsub
           is 1.
         - line_h: the height of each text line
         - lh: Alias for ``line_h``.
         - main_h: the input height
         - h: Alias for ``main_h``.
         - H: Alias for ``main_h``.
         - main_w: the input width
         - w: Alias for ``main_w``.
         - W: Alias for ``main_w``.
         - ascent: the maximum distance from the baseline to the highest/upper grid coordinate used to place a glyph
           outline point, for all the rendered glyphs. It is a positive value, due to the grid's orientation with the Y
           axis upwards.
         - max_glyph_a: Alias for ``ascent``.
         - descent: the maximum distance from the baseline to the lowest grid coordinate used to place a glyph outline
           point, for all the rendered glyphs. This is a negative value, due to the grid's orientation, with the Y axis
           upwards.
         - max_glyph_d: Alias for ``descent``.
         - max_glyph_h: maximum glyph height, that is the maximum height for all the glyphs contained in the rendered
           text, it is equivalent to ascent - descent.
         - max_glyph_w: maximum glyph width, that is the maximum width for all the glyphs contained in the rendered
           text.
         - n: the number of input frame, starting from 0
         - rand(min, max): return a random number included between min and max
         - sar: The input sample aspect ratio.
         - t: timestamp expressed in seconds, NAN if the input timestamp is unknown
         - text_h: the height of the rendered text
         - th: Alias for ``text_h``.
         - text_w: the width of the rendered text
         - tw: Alias for ``text_w``.
         - x: the x offset coordinates where the text is drawn.
         - y: the y offset coordinates where the text is drawn.

        These parameters allow the x and y expressions to refer each other, so you can for example specify
        ``y=x/dar``.

    Official documentation: `drawtext <https://ffmpeg.org/ffmpeg-filters.html#drawtext>`__
    """
    if text is not None:
        if escape_text:
            text = escape_chars(text, "\\'%")
        kwargs['text'] = text
    if x != 0:
        kwargs['x'] = x
    if y != 0:
        kwargs['y'] = y
    return filter(stream, drawtext.__name__, **kwargs)