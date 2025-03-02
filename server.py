from flask import Flask, request, jsonify
import os
from transformers import AutoProcessor, AutoModelForCausalLM, GenerationConfig
import torch
from PIL import Image
device = 'cuda:0'
model_path = "microsoft/Phi-4-multimodal-instruct"

# Load the custom model
class CustomModel:
    def __init__(self, model_path, language_only):
        self.model_path = model_path
        self.language_only = language_only
        self.model_type = 'custom'

        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            device_map=device, 
            torch_dtype=torch.bfloat16, 
            trust_remote_code=True, 
            attn_implementation="flash_attention_2"
        ).to(device)

        self.generation_config = GenerationConfig.from_pretrained(model_path)

    def respond(self, prompt, image_path=None):
        if 'microsoft/Phi-4' in self.model_path:
            user_prompt = '<|user|>'
            assistant_prompt = '<|assistant|>'
            prompt_suffix = '<|end|>'
            formatted_prompt = f'{user_prompt}<|image_1|>{prompt}{prompt_suffix}{assistant_prompt}'
            
            image = Image.open(image_path)
            inputs = self.processor(text=formatted_prompt, images=image, return_tensors='pt').to(self.model.device)
        else:
            raise NotImplementedError
        
        with torch.no_grad():
            generate_ids = self.model.generate(
                **inputs,
                max_new_tokens=1024,  # Adjust as needed
                temperature=0.0,      # Adjust as needed
                generation_config=self.generation_config,
            )
        
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        response = self.processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        # import pdb;pdb.set_trace()
        return response

# Initialize Flask app and model
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = CustomModel(model_path=model_path, language_only=False)

@app.route('/process', methods=['POST'])
def process_request():
    if 'image' not in request.files or 'sentence' not in request.form:
        return jsonify({'error': 'Missing image or sentence'}), 400

    image = request.files['image']
    sentence = request.form['sentence']

    if image.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save the image temporarily
    image_path = os.path.join(UPLOAD_FOLDER, image.filename)
    image.save(image_path)

    # Generate response from the model
    model_response = model.respond(sentence, image_path=image_path)

    return jsonify({'response': model_response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=23333)