from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import BatchEncoding
def post_process_video_grounding(self, logits, video_durations):
    """
        Compute the time of the video.

        Args:
            logits (`torch.Tensor`):
                The logits output of TvpForVideoGrounding.
            video_durations (`float`):
                The video's duration.

        Returns:
            start (`float`):
                The start time of the video.
            end (`float`):
                The end time of the video.
        """
    start, end = (round(logits.tolist()[0][0] * video_durations, 1), round(logits.tolist()[0][1] * video_durations, 1))
    return (start, end)