import os

import matplotlib.pyplot as plt
import torch
import torchdiffeq
from torchdyn.core import NeuralODE
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid
from tqdm import tqdm
import numpy as np

from torchcfm.conditional_flow_matching import *
from torchcfm.models.unet import UNetModel
from torchvision import datasets, transforms

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

def exponential_interval(a, b, n_points, exp_base=5.0) -> torch.Tensor:
    u = torch.linspace(0, 1, n_points)
    e_scaled = (exp_base ** u - 1) / (exp_base - 1)  # 标准化到 [0, 1]
    x = a + e_scaled * (b - a)
    return x

def integral(x_list, y_list, increase=True):
    # Compute the integral using the trapezoidal rule
    integral = 0.0
    for i in range(len(x_list) - 1):
        delta = (x_list[i + 1] - x_list[i]) * (y_list[i] + y_list[i + 1]) / 2
        if increase:
            integral += delta
        else:
            integral -= delta
    return integral

def compute_divergence(xt, t, cls, sample=10):
    vt = model.forward(t, xt, cls).view(-1, 1, 32, 32)
    div = np.zeros((xt.shape[0],), dtype=np.float32)
    # Approximate the Jacobian by Hutchinson’s Trace Estimator
    for _ in range(sample):  # Sample 10 times
        e = torch.randn_like(vt)
        dot = torch.sum(vt * e, dim=(1, 2, 3))
        grad = torch.autograd.grad(dot, xt, grad_outputs=torch.ones_like(dot), create_graph=False, retain_graph=True)[0]
        div += torch.sum(grad * e, dim=(1, 2, 3)).detach().cpu().numpy() / sample
    return div # Take the average of the samples

def classifier(x, steps=2, sample=10):
    # Enumerate the classes
    x = x.repeat(10, 1, 1, 1)  # Shape: (1, 1, 32, 32)
    classes = torch.arange(10, device=device)
    time_steps = exponential_interval(1, 0, steps, exp_base=100.0).to(device)  # Shape: (steps,)
    # Reverse flow
    with torch.no_grad():
        if USE_TORCH_DIFFEQ:
            traj = torchdiffeq.odeint(
                lambda t, x: model.forward(t, x, classes),
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
    
    log_prob = -torch.sum(init ** 2, dim=(1, 2, 3)) / 2 - 0.5 * 32 * 32 * torch.log(torch.tensor(2 * np.pi, device=device))
    log_prob = log_prob.detach().cpu().numpy()

    # Compute the divergence
    div_list = []
    for i in range(steps):
        t = time_steps[i]
        xt = traj[i].view(-1, 1, 32, 32)
        xt.requires_grad_(True)
        div = compute_divergence(xt, t, classes, sample=sample)
        div_list.append(div)
    print(f"div_list: {div_list}")
    # Integrate the divergence
    log_prob -= integral(time_steps.cpu().numpy(), div_list, increase=False)
        
    return np.argmax(log_prob), log_prob
    
if __name__ == "__main__":
    
    # Load MNIST dataset
    transform = transforms.Compose([transforms.Pad(2),transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    mnist = datasets.MNIST(root="data", train=True, download=True, transform=transform)

    # Select 10 sample image and label pairs
    sample_indices = np.random.choice(len(mnist), 100, replace=False)
    samples = [mnist[i] for i in sample_indices]
    acc_num = 0
    
    with tqdm(samples, desc="Processing samples") as pbar:
        for i, (img, label) in enumerate(pbar):
            img = img.unsqueeze(0).to(device)
            prediction, _ = classifier(img, steps=100, sample=20)
            if prediction == label:
                acc_num += 1
            pbar.set_postfix({"Prediction": prediction, "Label": label, "Accuracy": acc_num / (i + 1)})
            
        
        
        
        
        
        
        
        
