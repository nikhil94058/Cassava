from flask import Flask, request, jsonify
from PIL import Image
import io
from flask_cors import CORS
# Initialize Flask application
app = Flask(__name__)
CORS(app)
# Route to handle image upload and processing
@app.route('/upload', methods=['POST'])
def upload_image():
    # Check if the 'file' part is in the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    # Check if a file is selected
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Open the uploaded image
        image = Image.open(file.stream)

        # Here you can process the image if needed (e.g., classification, transformation, etc.)
        image.show()  # This will display the image

        return jsonify({'message': 'Image uploaded and processed successfully!'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
