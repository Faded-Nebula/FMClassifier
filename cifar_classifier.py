import os

import matplotlib.pyplot as plt
import torch
import torchdiffeq
from torchdyn.core import NeuralODE
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid
from tqdm import tqdm
import numpy as np
from collections import Counter

from torchcfm.conditional_flow_matching import *
from torchcfm.models.unet.unet import UNetModelWrapper
from torchvision import datasets, transforms

# Load model
USE_TORCH_DIFFEQ = True
savedir = "models/cond_cifar10"
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model = UNetModelWrapper(
    dim=(3, 32, 32),
    num_res_blocks=2,
    num_channels=128,
    channel_mult=[1, 2, 2, 2],
    class_cond=True,
    num_classes=10,
    num_heads=4,
    num_head_channels=64,
    attention_resolutions="16",
    dropout=0.1,
).to(device)
# use torch.load
PATH = f"./results/fm/fm_cifar10_weights_step_400000.pt"
print("path: ", PATH)
checkpoint = torch.load(PATH, map_location=device)
state_dict = checkpoint["ema_model"]
try:
    model.load_state_dict(state_dict)
except RuntimeError:
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_state_dict[k[7:]] = v
    model.load_state_dict(new_state_dict)
node = NeuralODE(model, solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4)

def exponential_interval(a, b, n_points, exp_base=5.0) -> torch.Tensor:
    u = torch.linspace(0, 1, n_points)
    e_scaled = (exp_base ** u - 1) / (exp_base - 1)  # 标准化到 [0, 1]
    x = a + e_scaled * (b - a)
    return x

def integral(x_list, y_list, increase=True, x_init=0.0, x_final=1.0):
    # Compute the integral using the trapezoidal rule
    integral = 0.0
    for i in range(len(x_list) - 1):
        if x_list[i] < x_init or x_list[i + 1] > x_final:
            continue
        delta = (x_list[i + 1] - x_list[i]) * (y_list[i] + y_list[i + 1]) / 2
        if increase:
            integral += delta
        else:
            integral -= delta
    return integral

def compute_divergence(xt, t, cls, sample=10, stochastic=False):
    if not stochastic:
        batch_size = xt.shape[0]

        # Repeat inputs for vectorized sampling
        xt_rep = xt.repeat(sample, 1, 1, 1)
        cls_rep = cls.repeat(sample)
        
        
        vt = model.forward(t, xt_rep, cls_rep).view(-1, 3, 32, 32)
        noise = torch.randn_like(vt, device=device)

        dot = torch.sum(vt * noise, dim=(1, 2, 3))
        grad_vt = torch.autograd.grad(
            outputs=dot,
            inputs=xt_rep,
            grad_outputs=torch.ones_like(dot),
            create_graph=False,
            retain_graph=False
        )[0]

        divergence = torch.sum(grad_vt * noise, dim=(1, 2, 3))
        divergence = divergence.view(sample, batch_size).mean(dim=0)
    else:
        batch_size = xt.shape[0]
        # Repeat inputs for vectorized sampling
        xt_rep = xt.repeat(sample, 1, 1, 1)
        cls_rep = cls.repeat(sample)

        # Use Rademacher distribution for noise
        noise = torch.randint(0, 2, xt_rep.shape, device=device, dtype=torch.float32) * 2 - 1
        xt_noisy = xt_rep + 1e-3 * noise

        vt = model.forward(t, xt_rep, cls_rep).view(-1, 3, 32, 32)
        vt_noisy = model.forward(t, xt_noisy, cls_rep).view(-1, 3, 32, 32)

        divergence = torch.sum((vt_noisy - vt) * noise, dim=(1, 2, 3)) / 1e-3
        divergence = divergence.view(sample, batch_size).mean(dim=0)
        
    return divergence.detach().cpu().numpy()

def classifier(x, steps=2, sample=10, stochastic=False):
    # Enumerate the classes
    x = x.repeat(10, 1, 1, 1)  # Shape: (1, 3, 32, 32)
    classes = torch.arange(10, device=device)
    time_steps = torch.linspace(1, 0, steps).to(device)  # Shape: (steps,)
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
    init = traj[-1].view(-1, 3, 32, 32)
    
    log_prob = -torch.sum(init ** 2, dim=(1, 2, 3)) / 2 - 0.5 * 3 * 32 * 32 * torch.log(torch.tensor(2 * np.pi, device=device))
    log_prob = log_prob.detach().cpu().numpy()

    # Compute the divergence
    div_list = []
    for i in range(steps):
        t = time_steps[i]
        xt = traj[i].view(-1, 3, 32, 32)
        xt.requires_grad_(True)
        div = compute_divergence(xt, t, classes, sample=sample, stochastic=stochastic)
        div_list.append(div)
        
    def log_prob_func(t):
        return log_prob - integral(time_steps.cpu().numpy(), div_list, increase=False, x_init=0.0, x_final=t)
        
    # Analyze the log prob from 0 to 1
    log_prob_time_list = [log_prob_func(t) for t in time_steps.cpu().numpy()]
    prediction = np.argmax(log_prob_time_list, axis=1)
        
    return prediction
    
if __name__ == "__main__":
    steps = 200
    
    # Load CIFAR-10 dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    cifar10 = datasets.CIFAR10(root="data", train=False, download=True, transform=transform)

    # Select 100 sample image and label pairs
    sample_indices = np.random.choice(len(cifar10), 100, replace=False)
    samples = [cifar10[i] for i in sample_indices]
    acc_num_time_list = [0 for _ in range(steps)]
    
    with tqdm(samples, desc="Processing samples") as pbar:
        for i, (img, label) in enumerate(pbar):
            img = img.unsqueeze(0).to(device)
            prediction_time_list = classifier(img, steps=steps, sample=5, stochastic=True)
            
            # Count the number of correct predictions at each time step
            acc_num_time_list = [acc_num_time_list[j] + (prediction_time_list[j] == label) for j in range(steps)]
            pbar.set_postfix({"Accuracy": np.max(acc_num_time_list) / (i + 1)})
    # Calculate the accuracy at each time step
    acc_time_list = [acc_num / len(samples) for acc_num in acc_num_time_list]
    # Plot the accuracy over time
    plt.plot(np.linspace(1, 0 , steps), acc_time_list)
    plt.xlabel("Time Step")
    plt.ylabel("Accuracy")
    plt.title("Classifier Accuracy Over Time")
    plt.grid()
    plt.savefig("classifier_accuracy_cifar10.png")