from flask import Flask, render_template, request, flash
import tensorflow as tf
import os
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # required for flash messages
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024  # Limit upload size to 1MB

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Load the trained model
#model = tf.keras.models.load_model('models2/transfer_model.keras')

model_path = os.path.join(os.path.dirname(__file__), 'models2', 'transfer_model.keras')
model = tf.keras.models.load_model(model_path)


class_names = ['French Bulldog', 'Siberian Husky']

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Helper function to prepare image for prediction
def prepare_image(image_path):
    print(f">>> [LOG] Preparing image: {image_path}")
    img = Image.open(image_path).convert('RGB')
    print(">>> [LOG] Image opened and converted to RGB")
    img = img.resize((224, 224))  # Resize to the input size expected by the model
    print(">>> [LOG] Image resized to 224x224")
    img_array = np.array(img) / 255.0
    print(">>> [LOG] Image converted to numpy array and normalized")
    img_array = np.expand_dims(img_array, axis=0)
    print(">>> [LOG] Image expanded to batch dimension")
    return img_array




# Home page
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        print(">>> [LOG] POST request received")
        file = request.files['file']
        if file and allowed_file(file.filename):
            print(">>> [LOG] File received and allowed")
            filename = secure_filename(file.filename)
            upload_folder = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
            filepath = os.path.join(upload_folder, filename)
            file.save(filepath)
            print(f">>> [LOG] File saved at {filepath}")

            try:
                print(">>> [LOG] Starting image preparation")
                img = prepare_image(filepath)
                print(">>> [LOG] Image prepared successfully")

                print(">>> [LOG] Starting model prediction")
                pred = model.predict(img)
                print(">>> [LOG] Prediction completed")

                confidence = np.max(pred)
                predicted_class = class_names[np.argmax(pred)]
                print(f">>> [LOG] Prediction result: {predicted_class} with confidence {confidence}")

                if confidence < 0.98:
                    prediction = "The uploaded image is neither a French Bulldog nor a Siberian Husky."
                else:
                    prediction = f"Breed: {predicted_class}"

            except Exception as e:
                flash('An error occurred while processing the image.')
                print(f">>> [ERROR] Exception occurred: {e}")

            finally:
                if os.path.exists(filepath):
                    os.remove(filepath)
                print(f">>> [LOG] Temporary file {filepath} removed")

        else:
            flash('Only .png, .jpg, and .jpeg files are allowed.')
            print(">>> [LOG] File not allowed")

    return render_template('index.html', prediction=prediction)


# Run the app
port = int(os.environ.get("PORT", 8080))
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port, debug=True)



