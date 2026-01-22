from typing import List, Optional, Tuple, Union
from ...image_processing_utils import BatchFeature
from ...image_transforms import center_to_corners_format
from ...image_utils import ImageInput
from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTokenizedInput, TextInput, TruncationStrategy
from ...utils import TensorType, is_torch_available
def post_process_grounded_object_detection(self, outputs, input_ids, box_threshold: float=0.25, text_threshold: float=0.25, target_sizes: Union[TensorType, List[Tuple]]=None):
    """
        Converts the raw output of [`GroundingDinoForObjectDetection`] into final bounding boxes in (top_left_x, top_left_y,
        bottom_right_x, bottom_right_y) format and get the associated text label.

        Args:
            outputs ([`GroundingDinoObjectDetectionOutput`]):
                Raw outputs of the model.
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The token ids of the input text.
            box_threshold (`float`, *optional*, defaults to 0.25):
                Score threshold to keep object detection predictions.
            text_threshold (`float`, *optional*, defaults to 0.25):
                Score threshold to keep text detection predictions.
            target_sizes (`torch.Tensor` or `List[Tuple[int, int]]`, *optional*):
                Tensor of shape `(batch_size, 2)` or list of tuples (`Tuple[int, int]`) containing the target size
                `(height, width)` of each image in the batch. If unset, predictions will not be resized.
        Returns:
            `List[Dict]`: A list of dictionaries, each dictionary containing the scores, labels and boxes for an image
            in the batch as predicted by the model.
        """
    logits, boxes = (outputs.logits, outputs.pred_boxes)
    if target_sizes is not None:
        if len(logits) != len(target_sizes):
            raise ValueError('Make sure that you pass in as many target sizes as the batch dimension of the logits')
    probs = torch.sigmoid(logits)
    scores = torch.max(probs, dim=-1)[0]
    boxes = center_to_corners_format(boxes)
    if target_sizes is not None:
        if isinstance(target_sizes, List):
            img_h = torch.Tensor([i[0] for i in target_sizes])
            img_w = torch.Tensor([i[1] for i in target_sizes])
        else:
            img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(boxes.device)
        boxes = boxes * scale_fct[:, None, :]
    results = []
    for idx, (s, b, p) in enumerate(zip(scores, boxes, probs)):
        score = s[s > box_threshold]
        box = b[s > box_threshold]
        prob = p[s > box_threshold]
        label_ids = get_phrases_from_posmap(prob > text_threshold, input_ids[idx])
        label = self.batch_decode(label_ids)
        results.append({'scores': score, 'labels': label, 'boxes': box})
    return results