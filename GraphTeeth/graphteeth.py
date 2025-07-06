import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from model.MEFL import *
from model.roi_heads import RoIHeads
from model.MEFL import MEFARG # use edge
# from model.MEFL_node import MEFARG # only node
 
class GraphPredictor(nn.Module):
    """
    Standard classification + bounding box regression layers
    for Fast R-CNN.

    Args:
        in_channels (int): number of input channels
        num_classes (int): number of output classes (including background)
    """

    def __init__(self, in_channels, box_head, num_classes, num_node):
        super().__init__()
        self.cls_score = MEFARG(num_node=num_node, num_classes=num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)
        self.box_head = box_head

    def forward(self, x):
        global_features, box_features = x

        # print(f"GraphPredictor input - Global: {type(global_features)}, Box: {box_features.shape}")

        # Store original box features for MEFARG (graph network)
        original_box_features = box_features.clone()

        # Apply box head transformation for bbox regression
        processed_box_features = self.box_head(box_features)

        # print(f"After box_head - Box features: {processed_box_features.shape}")

        # Flatten for bbox regression
        if processed_box_features.dim() == 4:
            torch._assert(
                list(processed_box_features.shape[2:]) == [1, 1],
                f"box_features has the wrong shape, expecting the last two dimensions to be [1,1] instead of {list(processed_box_features.shape[2:])}",
            )
        flattened_box_features = processed_box_features.flatten(start_dim=1)

        # print(f"Flattened box features: {flattened_box_features.shape}")

        # For MEFARG: use original features (rich spatial info)
        mefarg_input = (global_features, original_box_features)
        mefarg_output = self.cls_score(mefarg_input)

        # MEFARG returns (classification_logits, contrastive_embeddings)
        # For Fast R-CNN loss, we only need the classification logits
        if isinstance(mefarg_output, tuple):
            scores, contrastive_embeddings = mefarg_output
            # print(f"MEFARG output - Scores: {scores.shape}, Embeddings: {contrastive_embeddings.shape}")
        else:
            scores = mefarg_output
            # print(f"MEFARG output - Scores: {scores.shape}")

        # For bbox regression: use processed + flattened features
        bbox_deltas = self.bbox_pred(flattened_box_features)

        # print(f"Final output - Scores: {scores.shape}, BBox deltas: {bbox_deltas.shape}")

        return scores, bbox_deltas
  
class GraphTeeth(nn.Module):
    def __init__(self, num_classes=11, num_proposals=50, arc='resnet50'):
        super().__init__()
        
        backbone = resnet_fpn_backbone(
            backbone_name=arc,  
            weights='IMAGENET1K_V1' 
        )

        self.faster_rcnn = FasterRCNN(
            backbone=backbone,
            num_classes=num_classes,  
            rpn_post_nms_top_n_test=num_proposals  
        )
        in_features = self.faster_rcnn.roi_heads.box_predictor.cls_score.in_features
        box_predictor = GraphPredictor(in_features, self.faster_rcnn.roi_heads.box_head, num_classes, num_node=num_proposals)
        roi_heads = RoIHeads(
            self.faster_rcnn.roi_heads.box_roi_pool,
            box_head=None,
            box_predictor=box_predictor,
            # Faster R-CNN training
            fg_iou_thresh=0.5,
            bg_iou_thresh=0.5,
            batch_size_per_image=num_proposals,
            positive_fraction=0.25,
            bbox_reg_weights=None,
            # Faster R-CNN inference
            score_thresh=0.05,
            nms_thresh=0.5,
            detections_per_img=100
            )
        self.faster_rcnn.roi_heads = roi_heads


    def forward(self, images, targets=None):
        
        if self.training:
            loss_dict = self.faster_rcnn(images, targets)
            return loss_dict
        else:
            detection = self.faster_rcnn(images)
            return detection

