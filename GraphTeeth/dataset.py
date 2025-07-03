import numpy as np
import random
from PIL import Image
import os
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision import transforms
import conf

# ---------------- Contrastive Transforms ----------------
class TwoCropTransform:
    """Generate two tensor augmentations from a single PIL image."""
    def __init__(self, base_transform):
        self.base = base_transform  # expects PIL image

    def __call__(self, pil_img):
        # apply augmentations twice, returns two Tensors
        return self.base(pil_img), self.base(pil_img)

# Base augmentations for contrastive branch (PIL -> Tensor)
base_contrastive_augs = transforms.Compose([
    transforms.RandomResizedCrop((800, 1333), scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
    transforms.GaussianBlur(kernel_size=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
contrastive_transform = TwoCropTransform(base_contrastive_augs)

# Default transform for non-contrastive branch (PIL -> Tensor)
def get_default_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

class OMNIDataset(CocoDetection):
    def __init__(self, root, annFile,
                 train=True,
                 input_size=(1333, 800),
                 contrastive=False,
                 pair_transform=None):
        super().__init__(root, annFile)
        self.train = train
        self.input_size = input_size
        self.contrastive = contrastive
        self.pair_transform = pair_transform
        self.default_transform = get_default_transform()

    def _load_image(self, id: int) -> Image.Image:
        info = self.coco.loadImgs(id)[0]
        path = info["file_name"]
        sub1 = info['subfolder1']
        sub2 = info['subfolder2']
        full = os.path.join(self.root, sub1, sub2, path)
        return Image.open(full).convert("RGB")

    def __getitem__(self, idx):
        # load PIL image + coarse targets
        pil_img, target = super().__getitem__(idx)
        image_id = self.ids[idx]

        # parse and filter bboxes
        boxes, labels = [], []
        for obj in target:
            x, y, w, h = obj['bbox']
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(obj['category_id'])
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        # resize+pad and random flip on PIL image
        pil_img, boxes = self.resize_keep_ratio(pil_img, boxes)
        if self.train:
            pil_img, boxes = self.random_flip(pil_img, boxes)

        # assemble target dict
        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([image_id])}

        # contrastive branch: takes PIL image -> two augmented Tensors
        if self.contrastive and self.pair_transform:
            img1, img2 = self.pair_transform(pil_img)
            return (img1, img2), target

        # non-contrastive: single Tensor
        img_tensor = self.default_transform(pil_img)
        return img_tensor, target

    def resize_keep_ratio(self, image, bboxes):
        w, h = image.size
        new_w, new_h = self.input_size
        scale = min(new_w / w, new_h / h)
        resized = image.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
        canvas = Image.new("RGB", (new_w, new_h), (128, 128, 128))
        canvas.paste(resized, (0, 0))
        if len(bboxes) > 0:
            bboxes[:, [0, 2]] *= scale
            bboxes[:, [1, 3]] *= scale
        return canvas, bboxes

    def random_flip(self, image, bboxes, prob=0.5):
        if random.random() < prob:
            flipped = image.transpose(Image.FLIP_LEFT_RIGHT)
            img_w = flipped.size[0]
            if len(bboxes) > 0:
                bboxes[:, [0, 2]] = img_w - bboxes[:, [2, 0]]
            return flipped, bboxes
        return image, bboxes

# top-level collate_fn for contrastive batches
def collate_fn(batch):
    """
    Batch is a list of ((img1, img2), target) tuples.
    We stack img1 tensors and img2 tensors separately, and collect targets.
    """
    imgs, targets = zip(*batch)          # imgs: tuple of (img1, img2)
    x1_list, x2_list = zip(*imgs)        # separate lists
    x1_batch = torch.stack(x1_list, dim=0)
    x2_batch = torch.stack(x2_list, dim=0)
    return x1_batch, x2_batch, list(targets)

if __name__ == "__main__":
    root,annFile = conf.get_path()
    # instantiate contrastive dataset & loader
    ds = OMNIDataset(root, annFile,
                     train=True,
                     contrastive=True,
                     pair_transform=contrastive_transform)
    loader = DataLoader(ds,
                        batch_size=4,
                        shuffle=True,
                        num_workers=4,
                        collate_fn=collate_fn)
    for x1_batch, x2_batch, targets in loader:
        print(f"x1: {x1_batch.shape}, x2: {x2_batch.shape}")
        print(f"Number of targets: {len(targets)}")
        break
