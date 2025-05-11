import os
import torch
from torchvision import datasets, transforms
from torchcfm.conditional_flow_matching import *
from torchcfm.models.unet import UNetModel
from tqdm import tqdm

savedir = "models/cond_mnist"
os.makedirs(savedir, exist_ok=True)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(f"Using device: {device}")
batch_size = 128
n_epochs = 10

# Load MNIST
trainset = datasets.MNIST(
    "data",
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.Pad(2),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ]),
)
train_loader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True, drop_last=True
)

# Create model
sigma = 0.01
model = UNetModel(
    dim=(1, 32, 32),
    num_channels=32,
    num_res_blocks=2,
    channel_mult=(2, 2, 2, 2),
    num_classes=11,
    class_cond=True  # Required for CFG
).to(device)
optimizer = torch.optim.Adam(model.parameters())
FM = ConditionalFlowMatcher(sigma=sigma)

# Classifier-Free Guidance dropout probability
cfg_drop_prob = 0.1  # 10% of labels are dropped randomly during training

for epoch in range(n_epochs):
    with tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{n_epochs}") as pbar:
        for i, data in pbar:
            optimizer.zero_grad()
            x1 = data[0].to(device)
            y = data[1].to(device)

            # Classifier-Free Guidance: randomly drop labels
            mask = torch.rand(y.shape, device=device) < cfg_drop_prob
            y_cond = y.clone()
            y_cond[mask] = 10  # Special class index indicating "no label"

            x0 = torch.randn_like(x1)
            t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
            vt = model(t, xt, y_cond)  # Pass possibly dropped labels
            loss = torch.mean((vt - ut) ** 2)
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=loss.item())

# Save model
torch.save(model.state_dict(), os.path.join(savedir, "cfg.pth"))
