import torch
from torchvision import datasets, transforms
from PIL import Image
from diffusers import FluxPipeline
from huggingface_hub import login
import matplotlib.pyplot as plt
import numpy as np

# Hugging Face login
login(token="your_key_here")

# Load the FLUX pipeline
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16
)
pipe.enable_model_cpu_offload()

# CIFAR-10 class names
class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# Define preprocessing (resize to 1024x1024 to match FLUX if needed)
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Load 1 sample image from CIFAR-10 test set
dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
image_tensor, label = dataset[0]
image = image_tensor.unsqueeze(0).to(pipe.device, dtype=torch.bfloat16)

# Save the image
# Save the image as a PNG file for visualization
transforms.ToPILImage()(image_tensor).save("sample_image.png")

# Initialize log-probs as a list of lists for time-first ordering
num_steps = 100
log_probs = []  # [class][time]

for cls in class_names:
    prompt = f"A photo of a {cls}"
    log_prob_list, time_steps = pipe.reverseflow(
        image=image,
        prompt=prompt,
        height=32,
        width=32,
        guidance_scale=3.5,
        num_inference_steps=50,
        max_sequence_length=512,
        sample=10
    )
    # Convert scalar tensors to float and store the class's trajectory
    log_probs.append([log_prob.item() for log_prob in log_prob_list])

# Convert to numpy array of shape [class, time]
log_probs_np = np.array(log_probs)  # shape [C, T]
time_steps_np = time_steps.cpu().numpy() if isinstance(time_steps, torch.Tensor) else np.array(time_steps)

plt.figure(figsize=(10, 6))
for i in range(10):
    class_log_probs = log_probs_np[i]
    deviation = class_log_probs - log_probs_np.mean(axis=0)  # deviation from mean at each timestep
    plt.plot(time_steps_np, deviation, label=class_names[i])
plt.xlabel("Time Step")
plt.ylabel("Log Probability Deviation")
plt.title("Log Probability Deviation Over Time for Each Class")
plt.legend()
plt.tight_layout()
plt.savefig("log_prob_over_time.png")
plt.clf()

# Prediction: use the last time step
final_log_probs = log_probs_np[:, -1]
prediction = np.argmax(final_log_probs)
print(f"Predicted class: {class_names[prediction]}")
print(f"Truth: {class_names[label]}")



