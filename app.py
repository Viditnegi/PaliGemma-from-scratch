import os
import torch
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from inference import load_hf_model, PaliGemmaProcessor, test_inference

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads/"
app.config["MODEL_PATH"] = "paligemma-3b-pt-224"  

# Ensure the upload folder exists
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

device = "cpu"

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"

print("Device in use: ", device)

print(f"Loading model")
model, tokenizer = load_hf_model(app.config["MODEL_PATH"], device, half_precision=False)
model = model.to(device).eval()

# print(model)

num_image_tokens = model.config.vision_config.num_image_tokens
image_size = model.config.vision_config.image_size
processor = PaliGemmaProcessor(tokenizer, num_image_tokens, image_size)

def infer(
    prompt: str, 
    image_file_path: str = None, 
    max_tokens: int = 100,
    temperature: float = 0.8,
    top_p: float = 0.9,
    do_sample: bool = False,
):
    """Run inference using the preloaded model."""
    with torch.no_grad():
        output = test_inference(
            model,
            processor,
            device,
            prompt,
            image_file_path,
            max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
        )
    return output

@app.route('/', methods=['GET', 'POST'])
def index():
    output = None
    error_message = None
    image_path = None

    if request.method == 'POST':
        try:
            prompt = request.form.get("prompt", "")
            image_file = request.files.get("image")


            max_tokens_str = request.form.get("max_tokens_to_generate", "100")
            temperature_str = request.form.get("temperature", "0.8")
            top_p_str = request.form.get("top_p", "0.9")
            do_sample_str = request.form.get("do_sample", "false")

            try:
                max_tokens_to_generate = int(max_tokens_str)
                if max_tokens_to_generate < 0:
                    raise ValueError("max_tokens_to_generate must be >= 0.")
            except ValueError as e:
                error_message = f"Invalid value for max_tokens_to_generate: {max_tokens_str}. " \
                                f"Using default of 100. Error: {str(e)}"
                max_tokens_to_generate = 100

            try:
                temperature = float(temperature_str)
                if not (0 <= temperature <= 2):
                    raise ValueError("temperature must be between 0 and 2.")
            except ValueError as e:
                new_msg = f"Invalid value for temperature: {temperature_str}. " \
                          f"Using default of 0.8. Error: {str(e)}"
                error_message = f"{error_message}\n{new_msg}" if error_message else new_msg
                temperature = 0.8

            try:
                top_p = float(top_p_str)
                if not (0 <= top_p <= 1):
                    raise ValueError("top_p must be between 0 and 1.")
            except ValueError as e:
                new_msg = f"Invalid value for top_p: {top_p_str}. " \
                          f"Using default of 0.9. Error: {str(e)}"
                error_message = f"{error_message}\n{new_msg}" if error_message else new_msg
                top_p = 0.9

            do_sample = do_sample_str.lower() in ("true", "1", "yes", "on")

            if image_file and image_file.filename:
                filename = secure_filename(image_file.filename)
                image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                image_file.save(image_path)
            else:
                image_path = None

            output = infer(
                prompt=prompt,
                image_file_path=image_path,
                max_tokens=max_tokens_to_generate,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
            )

        except Exception as e:
            new_msg = f"An unexpected error occurred: {str(e)}"
            error_message = f"{error_message}\n{new_msg}" if error_message else new_msg

        finally:
            if image_path and os.path.exists(image_path):
                os.remove(image_path)

    return render_template("index.html", output=output, error=error_message)

if __name__ == '__main__':
    app.run(debug=False)
