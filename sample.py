import os
import zipfile
import tensorflow as tf
from flask import Flask, render_template, request, redirect, url_for, flash
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

app = Flask(__name__)
app.secret_key = "vitscan_secret_key"

# Configuration
UPLOAD_FOLDER = 'static/uploads/datasets'
MODEL_SAVE_PATH = 'static/models/oral_vitscan.h5'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('static/models', exist_ok=True)


def train_cnn_model(dataset_extract_path):
    """
    Core CNN Training Logic for Lip/Tongue classification.
    """
    # 1. Image Preprocessing (Normalizing and Augmenting)
    datagen = ImageDataGenerator(
        rescale=1. / 255,
        validation_split=0.2,  # 20% for testing
        rotation_range=20,
        horizontal_flip=True
    )

    train_generator = datagen.flow_from_directory(
        dataset_extract_path,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )

    # 2. Building the CNN Model (Simplified Architecture)
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(train_generator.num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # 3. Fit the Model (5 Epochs for Demo)
    model.fit(train_generator, epochs=5)

    # 4. Save for later prediction
    model.save(MODEL_SAVE_PATH)
    return True


@app.route('/train_dataset', methods=['GET', 'POST'])
def train_dataset():
    if request.method == 'POST':
        if 'dataset' not in request.files:
            flash("No file found!")
            return redirect(request.url)

        file = request.files['dataset']
        if file.filename == '':
            flash("No file selected!")
            return redirect(request.url)

        if file and file.filename.endswith('.zip'):
            # Save the ZIP
            zip_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(zip_path)

            # Extract the ZIP
            extract_folder = os.path.join(UPLOAD_FOLDER, file.filename.replace('.zip', ''))
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_folder)

            # Start Training
            try:
                train_cnn_model(extract_folder)
                flash("Model Retraining Successful! oral_vitscan.h5 updated.")
            except Exception as e:
                flash(f"Training Error: {str(e)}")

            return redirect(url_for('train_dataset'))

    return render_template('train_dataset.html')


if __name__ == '__main__':
    app.run(debug=True)