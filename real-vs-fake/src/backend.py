import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import AdamW
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import cv2

# Define the custom layer Patches before loading the model
class Patches(tf.keras.layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

# Define the custom layer PatchEncoder before loading the model
class PatchEncoder(tf.keras.layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = tf.keras.layers.Dense(units=projection_dim)
        self.position_embedding = tf.keras.layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

app = Flask(__name__)
CORS(app, supports_credentials=True, origins=['http://localhost:3000'])

# Function to create the AdamW optimizer
def create_adamw_optimizer(**kwargs):
    return tf.keras.optimizers.AdamW(**kwargs)

# Load the deepfake detection model, passing the custom optimizer function
custom_objects = {'Patches': Patches, 'PatchEncoder': PatchEncoder, 'AdamW': create_adamw_optimizer}
model = load_model('mymodel.h5', custom_objects=custom_objects,compile=False)

# Function to preprocess the image
def preprocess_image(image):
    # Convert image to numpy array
    img_array = np.array(image)
    # Convert BGR image to RGB
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    # Resize image to match model input size
    img_array = cv2.resize(img_array, (224, 224))
    # Normalize pixel values to range [0, 1]
    img_array = img_array / 255.0
    # Expand dimensions to create batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    return img_array
# Route for receiving image uploads and making predictions
@app.route('/detect', methods=['POST'])
def detect_deepfake():
    if 'image' not in request.files:
        return jsonify({'error': 'No image found in request'}), 400
    
    image = request.files['image']
    # Read the image using PIL
    img = Image.open(image)
    # Convert the image to a numpy array
    # img_array = np.array(img)
    # Preprocess the image
    processed_image = preprocess_image(img)
    # Make prediction using the loaded model
    predictions = model.predict(processed_image)
    # Convert prediction to a human-readable label
    prediction = predictions[0]
    # result = 'Real' if prediction < 0.5 else 'Fake'
    if np.any(predictions >= 0.5):
        result = 'Fake'
    else:
        result = 'Real'
    
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)
