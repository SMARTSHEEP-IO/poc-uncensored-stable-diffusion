from diffusers import StableDiffusionPipeline
import torch

device_type = "cpu"  # Can change to cuda for use of GPU

def load_model():
    model_id = "./models/nsfw"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32, safety_checker=None)

    pipe = pipe.to(device_type)
    return pipe

def generate_image(pipe, prompt, seed=None):
    generator = torch.Generator(device=device_type)

    if seed is not None:
        generator.manual_seed(seed)

    with torch.no_grad():
        image = pipe(prompt=prompt, num_inference_steps=50, guidance_scale=7.5, generator=generator).images[0]
    return image

def main():
    pipe = load_model()
    prompt = ("ultra realistic body of Ali Khamenei leader of Iran, "
              "He is in bed naked and feeling gay, hyper detail. The image should be "
              "hyper detailed, shot as if with a Canon EOS R3 and a Nikon lens at f/1.4, ISO 200, 1/160s, "
              "in 8K RAW format, unedited with a symmetrical balance.")

    # Set seed for reproducibility
    seed = 230908843746
    image = generate_image(pipe, prompt, seed)
    image.show()


if __name__ == "__main__":
    main()
