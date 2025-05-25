import torch
from diffusers import FluxPipeline
from torchvision.utils import make_grid
from torchvision.transforms import ToTensor, ToPILImage

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power

prompt = "A cat holding a sign that says hello world"
output = pipe(
    prompt,
    height=1024,
    width=1024,
    guidance_scale=0.5,
    num_inference_steps=50,
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(0),
)
# Flatten the images in case they are nested
trajectory = output.images
flattened_images = []
for item in trajectory:
    if isinstance(item, list):
        flattened_images.extend(item)
    else:
        flattened_images.append(item)

# Select 5 evenly spaced images
num_images = len(flattened_images)
selected_indices = torch.linspace(0, num_images - 1, steps=10, dtype=torch.long).tolist()
selected_images = [flattened_images[i] for i in selected_indices]

# Convert selected images to tensors
selected_images_tensors = [ToTensor()(img) for img in selected_images]

# Create a grid of the selected images
grid = make_grid(selected_images_tensors, nrow=10)

# Convert grid to PIL and save/show
grid_image = ToPILImage()(grid)
grid_image.save("trajectory_grid.png")
grid_image.show()
