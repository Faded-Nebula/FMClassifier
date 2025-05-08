import os

import matplotlib.pyplot as plt
import torch
import torchdiffeq
from torchdyn.core import NeuralODE
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid
from tqdm import tqdm

from torchcfm.conditional_flow_matching import *
from torchcfm.models.unet import UNetModel

# Load model
USE_TORCH_DIFFEQ = True
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

def classifier(x, steps=2):
    # Enumerate the classes
    x = x.repeat(10, 1, 1, 1) # Shape: (10, 1, 32, 32)
    classes = torch.arange(10, device=device)
    time_steps = torch.linspace(1, 0, steps, device=device)
    # Reverse flow
    with torch.no_grad():
        if USE_TORCH_DIFFEQ:
            traj = torchdiffeq.odeint(
                lambda t, x: -model.forward(t, x, classes),
                x,
                time_steps,
                atol=1e-4,
                rtol=1e-4,
                method="dopri5",
            )
        else:
            traj = node.trajectory(
                x,
                time_steps,
        )
    # Compute the initial log probability
    init = traj[-1].view(-1, 1, 32, 32)
    log_prob = -torch.sum(init ** 2, dim=(1, 2)) / 2
    
    # Average the divergence
    for i in range(steps):
        t = time_steps[i]
        xt = traj[i].view(-1, 1, 32, 32)
        # Compute the Jacobian
        jacobian = torch.autograd.functional.jacobian(
            lambda x: model.forward(t, x, classes),
            xt,
            create_graph=True,
            vectorize=True,
        ) # Shape: (B, 1, 32, 32, 1, 32, 32)
        jacobian = jacobian.view(-1, 1024, 1024) # Shape: (10, 1024, 1024)
        # Compute the divergence
        divergence = torch.stack([torch.trace(jacobian[idx]) for idx in range(jacobian.size(0))]) # Shape: (10,)
        log_prob -= divergence / steps
    print(log_prob)
    
if __name__ == "__main__":
    # Load a sample image
    x = torch.randn(1, 1, 32, 32, device=device)
    classifier(x, steps=2)
        
        
        
        
        
        
        
        
