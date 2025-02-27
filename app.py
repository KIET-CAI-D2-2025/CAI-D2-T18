from flask import Flask, render_template, request
from pymongo import MongoClient
from werkzeug.utils import secure_filename
from bson import Binary
import os

app = Flask(__name__)

# MongoDB connection setup
client = MongoClient("mongodb://localhost:27017")
db = client["audioDB"]
collection = db["audio_files"]

# Allowed file extensions
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'ogg'}

# Directory for saving uploaded files
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure the upload directory exists
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Check if the file type is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_audio():
    if request.method == 'POST':
        # Handle file upload
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return "No selected file", 400
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                # Save audio file to MongoDB
                with open(filepath, 'rb') as f:
                    file_data = f.read()
                    collection.insert_one({"filename": filename, "file": Binary(file_data)})
                
                return render_template('upload.html', success=True, filename=filename, mode='upload')

        # Handle audio recording
        elif 'audio_data' in request.form:
            audio_data = request.form['audio_data'].encode('utf-8')
            filename = 'recorded_audio.wav'
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            with open(filepath, 'wb') as f:
                f.write(audio_data)
            
            # Save recorded audio file to MongoDB
            with open(filepath, 'rb') as f:
                file_data = f.read()
                collection.insert_one({"filename": filename, "file": Binary(file_data)})
            
            return render_template('upload.html', success=True, filename=filename, mode='record')

    return render_template('upload.html')

if __name__ == "__main__":
    app.run(debug=True)