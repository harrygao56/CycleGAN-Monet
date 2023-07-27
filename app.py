from flask import Flask, jsonify, request, send_file
from flask_restful import Resource, Api, reqparse
from PIL import Image
import base64
from io import BytesIO
import torch
import numpy
from flask_cors import CORS
import numpy as np
from cyclegan.cyclegan import GeneratorResNet
import torchvision.transforms as transforms


# Create flask app
app = Flask(__name__)
CORS(app)
# Create flask api object
api = Api(app)

# Initializing parameters and other components
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_residual_layers = 19
input_shape = (3, 256, 256)
transforms_ = transforms.Compose([
    transforms.Resize((256, 256), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Initializing CycleGAN model
generator = GeneratorResNet(input_shape, num_residual_layers)
state_dict = torch.load("generator_G_final", map_location=device)
generator.load_state_dict(state_dict)
generator.eval()

# API generate painting endpoint
class Generate(Resource):
    # POST function to process image and generate painting
    def post(self):
        # Read input file
        file = request.files['file']
        file_content = file.read()

        if not file_content:
            return "fail"
        
        img = Image.open(BytesIO(file_content))

        # Save dimensions for transforming back to original size later
        og_dim = img.size

        # Transform the image and add a batch dimension
        img = transforms_(img).float().unsqueeze(0)

        # Feed image through CycleGAN
        painting = generator(img)

        # Convert image to correct format
        painting = np.transpose(painting.cpu().detach().numpy(), [0, 2, 3, 1])
        painting = painting / 2 + 0.5
        painting = (painting[0, :, :, :] * 255).astype(np.uint8)
        painting = Image.fromarray(painting).convert('RGB')

        # transform image back to original size (tuple needs to be flipped for some reason)
        post_transform = transforms.Resize(og_dim[::-1], Image.BICUBIC)
        painting = post_transform(painting)

        # Return image as b64 format
        rawBytes = BytesIO()
        painting.save(rawBytes, "JPEG")
        rawBytes.seek(0)
        painting_b64 = base64.b64encode(rawBytes.read())
        return jsonify({'painting': str(painting_b64)})


# Adding api endpoints
api.add_resource(Generate, '/generate')

if __name__ == '__main__':
    app.run(debug='true')