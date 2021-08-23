import torch
import torch.nn.functional as F
from network_files.losses_ import ghm_loss


def ghm_c_loss(inputs, targets):
    label_weight = torch.zeros_like(inputs, device=inputs.device)
    ghm_c = ghm_loss.GHMC()
    losses = ghm_c(inputs, targets, label_weight)
    return losses

def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2,
    reduction: str = "none",
):
    """
    Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py .
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples or -1 for ignore. Default = 0.25
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(
        inputs, targets, reduction="none"
    )
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss

def Iou_loss(preds: torch.Tensor,
             bbox: torch.Tensor,
             eps=1e-6,
             reduction='sum'):
    """
        Original implementation from https://blog.csdn.net/weixin_41803339/article/details/106372080
        preds:A tensor of shape [[x1, y1, x2, y2], [x1, y1, x2, y2], ...].
              The predictions for each example.
        bbox:A float tensor with the same shape as inputs.
             The ground truth box for each example.
        reduction:"mean"or"sum"
        return: iou_loss
    """
    x1 = torch.max(preds[:, 0], bbox[:, 0])
    y1 = torch.max(preds[:, 1], bbox[:, 1])
    x2 = torch.min(preds[:, 2], bbox[:, 2])
    y2 = torch.min(preds[:, 3], bbox[:, 3])

    w = (x2 - x1 + 1.0).clamp(0.)
    h = (y2 - y1 + 1.0).clamp(0.)
    inters = w * h

    uni = (preds[:, 2] - preds[:, 0] + 1.0) * (preds[:, 3] - preds[:, 1] + 1.0) + (
            bbox[:, 2] - bbox[:, 0] + 1.0) * (
                  bbox[:, 3] - bbox[:, 1] + 1.0) - inters
    ious = (inters / uni).clamp(min=eps)
    loss = -ious.log()

    if reduction == 'mean':
        loss = torch.mean(loss)
    elif reduction == 'sum':
        loss = torch.sum(loss)
    else:
        raise NotImplementedError
    return loss


def Giou_loss(preds: torch.Tensor,
              bbox: torch.Tensor,
              eps=1e-7,
              reduction='sum'):
    """
        Original implementation from https://blog.csdn.net/weixin_41803339/article/details/106372080
        https://github.com/sfzhang15/ATSS/blob/master/atss_core/modeling/rpn/atss/loss.py#L36
        preds:A tensor of shape [[x1, y1, x2, y2], [x1, y1, x2, y2], ...].
              The predictions for each example.
        bbox:A float tensor with the same shape as inputs.
             The ground truth box for each example.
        reduction:"mean"or"sum"
        return: giou_loss
    """
    ix1 = torch.max(preds[:, 0], bbox[:, 0])
    iy1 = torch.max(preds[:, 1], bbox[:, 1])
    ix2 = torch.min(preds[:, 2], bbox[:, 2])
    iy2 = torch.min(preds[:, 3], bbox[:, 3])

    iw = (ix2 - ix1 + 1.0).clamp(0.)
    ih = (iy2 - iy1 + 1.0).clamp(0.)

    # overlap
    inters = iw * ih
    # union
    uni = (preds[:, 2] - preds[:, 0] + 1.0) * (preds[:, 3] - preds[:, 1] + 1.0) + (bbox[:, 2] - bbox[:, 0] + 1.0) * (
            bbox[:, 3] - bbox[:, 1] + 1.0) - inters + eps
    # ious
    ious = inters / uni
    ex1 = torch.min(preds[:, 0], bbox[:, 0])
    ey1 = torch.min(preds[:, 1], bbox[:, 1])
    ex2 = torch.max(preds[:, 2], bbox[:, 2])
    ey2 = torch.max(preds[:, 3], bbox[:, 3])
    ew = (ex2 - ex1 + 1.0).clamp(min=0.)
    eh = (ey2 - ey1 + 1.0).clamp(min=0.)

    # enclose erea
    enclose = ew * eh + eps

    giou = ious - (enclose - uni) / enclose
    loss = 1 - giou

    if reduction == 'mean':
        loss = torch.mean(loss)
    elif reduction == 'sum':
        loss = torch.sum(loss)
    else:
        raise NotImplementedError
    return loss


def Diou_loss(preds: torch.Tensor,
              bbox: torch.Tensor,
              eps=1e-7,
              reduction='sum'):
    """
        Original implementation from https://blog.csdn.net/weixin_41803339/article/details/106372080
        preds:A tensor of shape [[x1, y1, x2, y2], [x1, y1, x2, y2], ...].
              The predictions for each example.
        bbox:A float tensor with the same shape as inputs.
             The ground truth box for each example.
        reduction:"mean"or"sum"
        return: diou_loss
    """
    ix1 = torch.max(preds[:, 0], bbox[:, 0])
    iy1 = torch.max(preds[:, 1], bbox[:, 1])
    ix2 = torch.min(preds[:, 2], bbox[:, 2])
    iy2 = torch.min(preds[:, 3], bbox[:, 3])

    iw = (ix2 - ix1 + 1.0).clamp(min=0.)
    ih = (iy2 - iy1 + 1.0).clamp(min=0.)

    # overlaps
    inters = iw * ih

    # union
    uni = (preds[:, 2] - preds[:, 0] + 1.0) * (preds[:, 3] - preds[:, 1] + 1.0) + (bbox[:, 2] - bbox[:, 0] + 1.0) * (
            bbox[:, 3] - bbox[:, 1] + 1.0) - inters

    # iou
    iou = inters / (uni + eps)

    # inter_diag
    cxpreds = (preds[:, 2] + preds[:, 0]) / 2
    cypreds = (preds[:, 3] + preds[:, 1]) / 2
    cxbbox = (bbox[:, 2] + bbox[:, 0]) / 2
    cybbox = (bbox[:, 3] + bbox[:, 1]) / 2

    inter_diag = (cxbbox - cxpreds) ** 2 + (cybbox - cypreds) ** 2

    # outer_diag
    ox1 = torch.min(preds[:, 0], bbox[:, 0])
    oy1 = torch.min(preds[:, 1], bbox[:, 1])
    ox2 = torch.max(preds[:, 2], bbox[:, 2])
    oy2 = torch.max(preds[:, 3], bbox[:, 3])

    outer_diag = (ox1 - ox2) ** 2 + (oy1 - oy2) ** 2
    diou = iou - inter_diag / outer_diag
    diou = torch.clamp(diou, min=-1.0, max=1.0)

    diou_loss = 1 - diou

    if reduction == 'mean':
        loss = torch.mean(diou_loss)
    elif reduction == 'sum':
        loss = torch.sum(diou_loss)
    else:
        raise NotImplementedError
    return loss

import math
def Ciou_loss(preds: torch.Tensor,
              bbox: torch.Tensor,
              eps=1e-7,
              reduction='sum'):
    """
        Original implementation from https://blog.csdn.net/weixin_41803339/article/details/106372080
        preds:A tensor of shape [[x1, y1, x2, y2], [x1, y1, x2, y2], ...].
              The predictions for each example.
        bbox:A float tensor with the same shape as inputs.
             The ground truth box for each example.
        reduction:"mean"or"sum"
        return: ciou_loss
    """
    ix1 = torch.max(preds[:, 0], bbox[:, 0])
    iy1 = torch.max(preds[:, 1], bbox[:, 1])
    ix2 = torch.min(preds[:, 2], bbox[:, 2])
    iy2 = torch.min(preds[:, 3], bbox[:, 3])

    iw = (ix2 - ix1 + 1.0).clamp(min=0.)
    ih = (iy2 - iy1 + 1.0).clamp(min=0.)

    # overlaps
    inters = iw * ih

    # union
    uni = (preds[:, 2] - preds[:, 0] + 1.0) * (preds[:, 3] - preds[:, 1] + 1.0) + (bbox[:, 2] - bbox[:, 0] + 1.0) * (
            bbox[:, 3] - bbox[:, 1] + 1.0) - inters

    # iou
    iou = inters / (uni + eps)

    # inter_diag
    cxpreds = (preds[:, 2] + preds[:, 0]) / 2
    cypreds = (preds[:, 3] + preds[:, 1]) / 2

    cxbbox = (bbox[:, 2] + bbox[:, 0]) / 2
    cybbox = (bbox[:, 3] + bbox[:, 1]) / 2

    inter_diag = (cxbbox - cxpreds) ** 2 + (cybbox - cypreds) ** 2

    # outer_diag
    ox1 = torch.min(preds[:, 0], bbox[:, 0])
    oy1 = torch.min(preds[:, 1], bbox[:, 1])
    ox2 = torch.max(preds[:, 2], bbox[:, 2])
    oy2 = torch.max(preds[:, 3], bbox[:, 3])

    outer_diag = (ox1 - ox2) ** 2 + (oy1 - oy2) ** 2

    diou = iou - inter_diag / outer_diag

    # calculate v,alpha
    wbbox = bbox[:, 2] - bbox[:, 0] + 1.0
    hbbox = bbox[:, 3] - bbox[:, 1] + 1.0
    wpreds = preds[:, 2] - preds[:, 0] + 1.0
    hpreds = preds[:, 3] - preds[:, 1] + 1.0
    v = torch.pow((torch.atan(wbbox / hbbox) - torch.atan(wpreds / hpreds)), 2) * (4 / (math.pi ** 2))
    alpha = v / (1 - iou + v)
    ciou = diou - alpha * v
    ciou = torch.clamp(ciou, min=-1.0, max=1.0)

    ciou_loss = 1 - ciou
    if reduction == 'mean':
        loss = torch.mean(ciou_loss)
    elif reduction == 'sum':
        loss = torch.sum(ciou_loss)
    else:
        raise NotImplementedError
    return loss



if __name__ == "__main__":
    pred_box = torch.tensor([[2, 4, 6, 8], [5, 9, 13, 12]])
    gt_box = torch.tensor([[3, 4, 7, 9], [5, 6, 15, 13]])
    loss = Ciou_loss(preds=pred_box, bbox=gt_box)
    print('Ciou_loss:', loss)
    loss = Iou_loss(preds=pred_box, bbox=gt_box)
    print('Iou_loss', loss)
    loss = Diou_loss(preds=pred_box, bbox=gt_box)
    print('Diou_loss', loss)
    loss = Giou_loss(preds=pred_box, bbox=gt_box)
    print('Giou_loss', loss)

# 输出结果
"""
iou:
 tensor([0.5714, 0.0476])
diou:
 tensor([0.5714, 0.0476])
last_loss:
 tensor(0.6940)
"""




