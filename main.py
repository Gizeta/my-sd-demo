import time
import numpy as np
from diffusers import OnnxStableDiffusionPipeline
from diffusers import (
    DDPMScheduler,
    DDIMScheduler,
    PNDMScheduler,
    LMSDiscreteScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
)

model_folder = "./models_onnx"
scheduler_name = "EulerDiscreteScheduler"

prompt = "a photo of an astronaut riding a horse on mars"
negative_prompt = "bad hands, blurry"
num_inference_steps = 28
guidance_scale = 7.5
width = 512
height = 512
seed = np.random.randint(1, 2147483647)

if scheduler_name == "DDPMScheduler":
  scheduler = DDPMScheduler.from_pretrained(model_folder, subfolder="scheduler", revision="onnx")
elif scheduler_name == "DDIMScheduler":
  scheduler = DDIMScheduler.from_pretrained(model_folder, subfolder="scheduler", revision="onnx")
elif scheduler_name == "PNDMScheduler":
  scheduler = PNDMScheduler.from_pretrained(model_folder, subfolder="scheduler", revision="onnx")
elif scheduler_name == "LMSDiscreteScheduler":
  scheduler = LMSDiscreteScheduler.from_pretrained(model_folder, subfolder="scheduler", revision="onnx")
elif scheduler_name == "EulerAncestralDiscreteScheduler":
  scheduler = EulerAncestralDiscreteScheduler.from_pretrained(model_folder, subfolder="scheduler", revision="onnx")
elif scheduler_name == "EulerDiscreteScheduler":
  scheduler = EulerDiscreteScheduler.from_pretrained(model_folder, subfolder="scheduler", revision="onnx")
elif scheduler_name == "DPMSolverMultistepScheduler":
  scheduler = DPMSolverMultistepScheduler.from_pretrained(model_folder, subfolder="scheduler", revision="onnx")
else:
  scheduler = DDPMScheduler.from_pretrained(model_folder, subfolder="scheduler", revision="onnx")

pipe = OnnxStableDiffusionPipeline.from_pretrained(
  "./models_onnx",
  revision="onnx",
  provider="DmlExecutionProvider",
  safety_checker=None,
  scheduler=scheduler,
)

gen_time = time.strftime("%m%d%Y-%H%M%S")
rng = np.random.RandomState(seed)
image = pipe(prompt, height, width, num_inference_steps, guidance_scale, negative_prompt, generator = rng).images[0]

image.save("./outputs/" + gen_time + ".png")
with open("./outputs/" + gen_time + ".txt", "a+", encoding="utf-8") as f:
  f.write("prompt: " + prompt + "\n")
  f.write("negative_prompt: " + negative_prompt + "\n")
  f.write("num_inference_steps: " + str(num_inference_steps) + "\n")
  f.write("guidance_scale: " + str(guidance_scale) + "\n")
  f.write("seed: " + str(seed) + "\n")
