import time
import numpy as np
from diffusers import OnnxStableDiffusionPipeline, OnnxRuntimeModel
from diffusers import (
    DDPMScheduler,
    DDIMScheduler,
    PNDMScheduler,
    LMSDiscreteScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
)

model_folder = "./models/anything-v4.0"
scheduler_name = "EulerAncestralDiscreteScheduler"

prompt = "best quality, illustration, extremely detailed CG unity 8k wallpaper, ultra-detailed, 1girl with long brown hair blown by wind wearing blue dress standing on the balcony with iron fence, falling leaves blown by wind, dark clouds, grey sky, beautiful detailed eyes, elegant eyelashes, beautiful detailed face, extremely detailed wallpaper, hair hangs down to the shoulder, eyes visible through hair"
negative_prompt = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, bad feet"
steps = 30
scale = 7.5
width = 512
height = 768
seed = np.random.randint(1, 2147483647)

if scheduler_name == "DDPMScheduler":
    scheduler = DDPMScheduler.from_pretrained(
        model_folder, subfolder="scheduler", revision="onnx")
elif scheduler_name == "DDIMScheduler":
    scheduler = DDIMScheduler.from_pretrained(
        model_folder, subfolder="scheduler", revision="onnx")
elif scheduler_name == "PNDMScheduler":
    scheduler = PNDMScheduler.from_pretrained(
        model_folder, subfolder="scheduler", revision="onnx")
elif scheduler_name == "LMSDiscreteScheduler":
    scheduler = LMSDiscreteScheduler.from_pretrained(
        model_folder, subfolder="scheduler", revision="onnx")
elif scheduler_name == "EulerAncestralDiscreteScheduler":
    scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
        model_folder, subfolder="scheduler", revision="onnx")
elif scheduler_name == "EulerDiscreteScheduler":
    scheduler = EulerDiscreteScheduler.from_pretrained(
        model_folder, subfolder="scheduler", revision="onnx")
elif scheduler_name == "DPMSolverMultistepScheduler":
    scheduler = DPMSolverMultistepScheduler.from_pretrained(
        model_folder, subfolder="scheduler", revision="onnx")
else:
    scheduler = DPMSolverMultistepScheduler.from_pretrained(
        model_folder, subfolder="scheduler", revision="onnx")
cputextenc = OnnxRuntimeModel.from_pretrained(model_folder + "/text_encoder")
cpuvae = OnnxRuntimeModel.from_pretrained(model_folder + "/vae_decoder")
pipe = OnnxStableDiffusionPipeline.from_pretrained(
    model_folder,
    revision="onnx",
    provider="DmlExecutionProvider",
    scheduler=scheduler,
    text_encoder=cputextenc,
    vae_decoder=cpuvae,
    vae_encoder=None,
)

gen_time = time.strftime("%m%d%Y-%H%M%S")
rng = np.random.RandomState(seed)
image = pipe(
    prompt=prompt,
    height=height,
    width=width,
    num_inference_steps=steps,
    guidance_scale=scale,
    negative_prompt=negative_prompt,
    num_images_per_prompt=1,
    generator=rng,
).images[0]

image.save("./outputs/" + gen_time + ".png")
with open("./outputs/" + gen_time + ".txt", "a+", encoding="utf-8") as f:
    f.write("prompt: " + prompt + "\n")
    f.write("negative_prompt: " + negative_prompt + "\n")
    f.write("num_inference_steps: " + str(steps) + "\n")
    f.write("guidance_scale: " + str(scale) + "\n")
    f.write("seed: " + str(seed) + "\n")
