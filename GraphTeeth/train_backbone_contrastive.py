# train_backbone_contrastive.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.models import resnet34

# Import your contrastive dataset and collate_fn
from dataset import OMNIDataset, contrastive_transform, collate_fn


# ---------------- NT-Xent Loss ----------------
def nt_xent_loss(z_i, z_j, temperature=0.5):
    """
    Compute NT-Xent loss for two batches of normalized embeddings.
    z_i, z_j: [N, D] where N = batch_size
    """
    N = z_i.size(0)
    # Concatenate embeddings
    z = torch.cat([z_i, z_j], dim=0)  # [2N, D]
    # Compute cosine similarity matrix
    sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)  # [2N,2N]
    # Mask self-similarities
    mask = (~torch.eye(2 * N, device=sim.device).bool()).float()
    exp_sim = torch.exp(sim / temperature) * mask
    denom = exp_sim.sum(dim=1)  # [2N]

    # Positive similarities: pairs (i, i+N) and (i+N, i)
    pos = torch.exp(torch.sum(z_i * z_j, dim=1) / temperature)
    pos = torch.cat([pos, pos], dim=0)  # [2N]

    # NT-Xent loss
    loss = -torch.log(pos / denom)
    return loss.mean()


# ---------------- ContrastiveResNet Model ----------------
class ContrastiveResNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Use ResNet-34 to match your GraphTeeth model
        self.backbone = resnet34(pretrained=False)  # Change this line
        self.backbone.fc = nn.Identity()

        # Projection head for ResNet-34 (512 features, same as ResNet-18)
        self.proj_head = nn.Sequential(
            nn.Linear(512, 256, bias=False),  # ResNet-34 has 512 features
            nn.ReLU(inplace=True),
            nn.Linear(256, 128)
        )

    def forward(self, x):
        # Extract features
        h = self.backbone(x)  # [B, 2048]
        # Project to contrastive embedding
        z = self.proj_head(h)  # [B, 128]
        # L2 normalization
        return F.normalize(z, dim=1)


# ---------------- Main ----------------
def main():
    # 1. Dataset & DataLoader
    dataset = OMNIDataset(
        root=r"D:\MScPro\OMNI_New\data\OMNI_COCO\testdata\train",
        annFile=r"D:\MScPro\OMNI_New\data\OMNI_COCO\testdata\annotations\instances_train.json",
        train=True,
        contrastive=True,
        pair_transform=contrastive_transform
    )
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn
    )

    # 2. Model & Optimizer
    model = ContrastiveResNet().cuda()
    optimizer = Adam(model.parameters(), lr=3e-4, weight_decay=1e-6)

    # 3. Training Loop
    epochs = 5
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for x1, x2, _ in loader:
            x1 = x1.cuda(non_blocking=True)
            x2 = x2.cuda(non_blocking=True)

            # Forward pass
            z1 = model(x1)  # [B, 128]
            z2 = model(x2)
            # Compute NT-Xent loss
            loss = nt_xent_loss(z1, z2, temperature=0.5)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x1.size(0)

        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch:03d} | Contrastive Loss: {avg_loss:.4f}")

    # 4. Save pretrained backbone weights
    torch.save(model.backbone.state_dict(), "backbone_contrastive2.pth")
    print("Saved contrastively pretrained backbone to backbone_contrastive.pth")


if __name__ == "__main__":
    main()