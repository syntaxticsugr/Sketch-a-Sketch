from PIL import Image
from controlnet_aux import HEDdetector
from diffusers import (
    ControlNetModel,
    EulerAncestralDiscreteScheduler,
    StableDiffusionControlNetPipeline
)
import gradio as gr
import numpy as np
import torch



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

controlnet = ControlNetModel.from_pretrained(
    'vsanimator/sketch-a-sketch'
).to(device)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    'runwayml/stable-diffusion-v1-5',
    controlnet=controlnet
).to(device)

pipe.safety_checker = None
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

hed = HEDdetector.from_pretrained(
    'lllyasviel/Annotators'
).to(device)



num_images = 3

def sketch(prompt, negative_prompt, curr_sketch, seed, num_steps):

    curr_sketch = curr_sketch if curr_sketch is not None else np.full((512, 512, 3), 255)

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    curr_sketch_image = Image.fromarray(curr_sketch.astype(np.uint8)).convert('L')

    images = pipe(
        prompt,
        curr_sketch_image.convert('RGB').point(lambda p: 256 if p > 128 else 0),
        negative_prompt=negative_prompt,
        num_inference_steps=num_steps,
        generator=generator,
        controlnet_conditioning_scale=1.0
    ).images

    return images[0]

def run_sketching(prompt, negative_prompt, curr_sketch, sketch_states):
    to_return = []

    curr_sketch = curr_sketch['composite']

    for k in range(num_images):
        seed = sketch_states[k][1] or np.random.randint(1000)
        sketch_states[k][1] = seed

        new_image = sketch(prompt, negative_prompt, curr_sketch, seed=seed, num_steps=20)
        to_return.append(new_image)

    curr_sketch = curr_sketch if curr_sketch is not None else np.full((512, 512, 3), 255)

    hed_images = [hed(image, scribble=False) for image in to_return]
    avg_hed = np.mean([np.array(image) for image in hed_images], axis = 0)
    curr_sketch = np.array(curr_sketch).astype(float) / 255.
    curr_sketch = Image.fromarray(np.uint8(1.0*((0.0*curr_sketch + 1. - 1.*(avg_hed / 255.))) * 255.))

    return to_return + [curr_sketch, sketch_states]

def reset(sketch_states):
    sketch_states = [[None, None] for _ in range(num_images)]
    return np.full((512, 512, 3), 255, dtype=np.uint8), sketch_states



with gr.Blocks(title="Sketch-a-Sketch Demo") as demo:
    start_state = [[None, None] for _ in range(num_images)]
    sketch_states = gr.State(start_state)

    with gr.Row():
        sketch_editor = gr.ImageEditor(
            label="Sketch Editor",
            height=800,
            type='numpy',
            image_mode='RGB',
            value=np.full((512, 512, 3), 255, dtype=np.uint8),
            brush=gr.Brush(
                default_color='black',
                default_size=2
            )
        )
        with gr.Column():
            prompt_box = gr.Textbox(label="Prompt")
            negative_prompt_box = gr.Textbox(label="Negative Prompt")
            with gr.Row():
                render_button = gr.Button("Render", variant='primary')
                reset_button = gr.Button("Reset", variant='stop')
            suggested_lines = gr.Image(label="Suggested Lines", height=550)

    with gr.Row():
        output_images = [gr.Image(label=f"Output Image {i+1}", height=500) for i in range(num_images)]

    render_button.click(
        run_sketching,
        inputs=[prompt_box, negative_prompt_box, sketch_editor, sketch_states],
        outputs=output_images + [suggested_lines, sketch_states]
    )

    reset_button.click(reset, inputs=sketch_states, outputs=[sketch_editor, sketch_states])



demo.launch(debug=True, inbrowser=True)
