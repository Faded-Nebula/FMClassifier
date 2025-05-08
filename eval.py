import os

import matplotlib.pyplot as plt
import torch
import torchdiffeq
from torchdyn.core import NeuralODE
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid

from torchcfm.conditional_flow_matching import *
from torchcfm.models.unet import UNetModel

USE_TORCH_DIFFEQ = True
# Load model
savedir = "models/cond_mnist"
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model = UNetModel(
    dim=(1, 32, 32), num_channels=32, num_res_blocks=2, channel_mult=(2, 2, 2, 2), num_classes=10, class_cond=True
)
# use torch.load
model.load_state_dict(
    torch.load(os.path.join(savedir, "model.pth"), map_location=device, weights_only=True), strict=False
)
model.to(device)
node = NeuralODE(model, solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4)

# Display generation results
def display_generation(model, steps=2):
    generated_class_list = torch.arange(10, device=device).repeat(10)
    with torch.no_grad():
        if USE_TORCH_DIFFEQ:
            traj = torchdiffeq.odeint(
                lambda t, x: model.forward(t, x, generated_class_list),
                torch.randn(100, 1, 32, 32, device=device),
                torch.linspace(0, 1, steps, device=device),
                atol=1e-4,
                rtol=1e-4,
                method="dopri5",
            )
        else:
            traj = node.trajectory(
                torch.randn(100, 1, 32, 32, device=device),
                t_span=torch.linspace(0, 1, steps, device=device),
            )
    grid = make_grid(
        traj[-1, :100].view([-1, 1, 32, 32])[:, :, 2:30, 2:30].clip(-1, 1), value_range=(-1, 1), padding=0, nrow=10
    )
    img = ToPILImage()(grid)
    plt.imshow(img)
    plt.savefig("generated_mnist.png")

if __name__ == "__main__":
    display_generation(model, steps=100)
    
