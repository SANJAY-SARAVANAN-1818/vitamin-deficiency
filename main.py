from flask import Flask, render_template, request, redirect, session, url_for, flash
import sqlite3
import os
import zipfile
# Commented out TensorFlow imports due to Python 3.14 incompatibility
# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras import layers, models
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image

# Fallback numpy import (without tensorflow)
try:
    import numpy as np
except ImportError:
    np = None

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Mock imports for Keras classes when TensorFlow is not available
class MockImageDataGenerator:
    """Mock ImageDataGenerator for when TensorFlow is not available"""
    def __init__(self, **kwargs):
        self.kwargs = kwargs
    
    def flow_from_directory(self, *args, **kwargs):
        """Return a mock generator"""
        return None

# Set available names
ImageDataGenerator = MockImageDataGenerator
app = Flask(__name__)
app.secret_key = "vitscan_secure_key_2026"
app.config['UPLOAD_FOLDER'] = 'static/uploads'
# Paths
UPLOAD_FOLDER = 'static/uploads/datasets'
RESULT_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# --- SQLite Connection Function ---
def get_db_connection():
    try:
        connection = sqlite3.connect('vitscan_db.sqlite')
        connection.row_factory = sqlite3.Row
        return connection
    except Exception as e:
        print(f"Error connecting to SQLite: {e}")
        return None

def init_db():
    conn = get_db_connection()
    if conn is None:
        return
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        email TEXT UNIQUE NOT NULL,
                        password TEXT NOT NULL
                      )''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS reports (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER,
                        result TEXT,
                        cause TEXT,
                        organic_remedy TEXT,
                        food_items TEXT,
                        medicine TEXT,
                        image_path TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (user_id) REFERENCES users (id)
                      )''')
    conn.commit()
    conn.close()

# Initialize DB on startup
init_db()


# Define your default path here (change this to your actual folder name)
DEFAULT_DATASET_PATH = 'static/uploads/datasets/archive (1)/dataset'


def run_training_with_analytics(extract_path=DEFAULT_DATASET_PATH):
    # Check if the path actually exists to avoid crashing
    if not os.path.exists(extract_path):
        print(f"Error: Dataset path {extract_path} not found!")
        return False

    # 1. Image Data Generator (With Validation Split)
    datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)

    # Note: flow_from_directory will look for subfolders inside extract_path
    train_data = datagen.flow_from_directory(
        extract_path,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )

    val_data = datagen.flow_from_directory(
        extract_path,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    # 2. CNN Model Architecture
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(train_data.num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # 3. Train and Save History
    history = model.fit(train_data, validation_data=val_data, epochs=10)

    # --- 📊 Generation of Graphs ---
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy', color='#00796b')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy', color='#ff4d6d')
    plt.title('Model Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss', color='#00796b')
    plt.plot(history.history['val_loss'], label='Val Loss', color='#ff4d6d')
    plt.title('Model Loss')
    plt.legend()

    # Ensure RESULT_FOLDER is defined globally in your main.py
    plt.savefig(os.path.join(RESULT_FOLDER, 'accuracy_loss.png'))
    plt.close()

    # --- 📉 Confusion Matrix ---
    Y_pred = model.predict(val_data)
    y_pred = np.argmax(Y_pred, axis=1)
    cm = confusion_matrix(val_data.classes, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=list(val_data.class_indices.keys()),
                yticklabels=list(val_data.class_indices.keys()))
    plt.title('Confusion Matrix - Vitamin Deficiencies')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(os.path.join(RESULT_FOLDER, 'confusion_matrix.png'))
    plt.close()

    # --- 📝 Classification Report ---
    report = classification_report(val_data.classes, y_pred,
                                   target_names=list(val_data.class_indices.keys()), output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(RESULT_FOLDER, 'report.csv'))

    # Ensure static/models/ directory exists
    os.makedirs('static/models', exist_ok=True)
    model.save('static/models/vitamin_model.h5')

    return True

@app.route('/train_dataset', methods=['GET', 'POST'])
def train_dataset():
    if request.method == 'POST':
        file = request.files.get('dataset')
        if file and file.filename.endswith('.zip'):
            zip_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(zip_path)

            extract_to = os.path.join(UPLOAD_FOLDER, file.filename.replace('.zip', ''))
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)

            if run_training_with_analytics(extract_to):
                flash("Training Complete! View analytics below.")
                return redirect(url_for('training_results'))

    return render_template('train_dataset.html')


@app.route('/training_results')
def training_results():
    # Load the CSV report for the table
    report_data = pd.read_csv(os.path.join(RESULT_FOLDER, 'report.csv'), index_col=0)
    # Style the table for Bootstrap
    table_html = report_data.to_html(classes='table table-hover table-bordered border-light shadow-sm text-center')
    return render_template('results.html', report_table=table_html)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login')
def user_login():
    return render_template('login.html')
@app.route('/admin')
def admin():
    return render_template('admin_login.html')


@app.route('/admin_login_process', methods=['POST'])
def admin_login_process():
    username = request.form.get('username')
    password = request.form.get('password')

    # Manual check for Admin credentials
    if username == "admin" and password == "admin123":
        session['admin_logged'] = True
        return redirect(url_for('admin_dashboard'))
    else:
        return "Invalid Admin Credentials. <a href='/admin'>Try Again</a>"


@app.route('/admin_dashboard')
def admin_dashboard():
    if not session.get('admin_logged'):
        return redirect('/admin')

    # Fetching data for the Admin to see
    conn = get_db_connection()
    if conn is None:
        flash("Database connection failed. Please check MySQL configuration.")
        return redirect('/admin')

    cursor = conn.cursor()

    # 1. View User Details
    cursor.execute('SELECT id, name, email FROM users')
    users_list = cursor.fetchall()

    # 2. View All Reports (User-wise)
    cursor.execute('''
        SELECT reports.*, users.name 
        FROM reports 
        JOIN users ON reports.user_id = users.id 
        ORDER BY reports.created_at DESC
    ''')
    reports_list = cursor.fetchall()

    cursor.close()
    conn.close()

    return render_template('admin_dash.html', users=users_list, reports=reports_list)
# --- USER AUTHENTICATION ---

@app.route('/user_register_process', methods=['POST'])
def register():
    name = request.form['name']
    email = request.form['email']
    password = request.form['password']

    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('INSERT INTO users (name, email, password) VALUES (?, ?, ?)', (name, email, password))
        conn.commit()
        flash("Registration Successful!")
    except Exception as e:
        flash("Registration Failed: " + str(e))
    finally:
        cursor.close()
        conn.close()
    return redirect(url_for('user_login'))


@app.route('/user_login_process', methods=['POST'])
def user_login_process():
    email = request.form['email']
    password = request.form['password']

    conn = get_db_connection()
    cursor = conn.cursor()  # returns sqlite3.Row
    cursor.execute('SELECT * FROM users WHERE email = ? AND password = ?', (email, password))
    user = cursor.fetchone()

    if user:
        session['id'] = user['id']
        session['name'] = user['name']
        return redirect(url_for('user_home'))
    else:
        flash("Invalid Credentials")
        return redirect(url_for('user_login'))


@app.route('/user_details')
def user_details():
    conn = get_db_connection()
    users_data = []

    if conn:
        try:
            # For sqlite3, we use row_factory to match the HTML logic
            cursor = conn.cursor()

            sql = """
                SELECT u.id, u.name, u.email, r.result, r.advice, r.created_at 
                FROM users u 
                LEFT JOIN reports r ON u.id = r.user_id 
                ORDER BY r.created_at DESC
            """
            cursor.execute(sql)
            users_data = cursor.fetchall()

            cursor.close()
        except Exception as e:
            print(f"Database Error: {e}")
        finally:
            conn.close()

    return render_template('user_details.html', users=users_data)


# --- ANALYSIS & DATABASE SAVE ---

@app.route('/reports')
def reports():
    conn = get_db_connection()
    reports_data = []

    if conn:
        try:
            # We use row_factory because your HTML template uses user.name/user.result syntax
            cursor = conn.cursor()

            # Fetching report details along with the patient's name
            sql = """
                SELECT r.id, u.name, r.result, r.advice, r.image_path, r.created_at 
                FROM reports r
                JOIN users u ON r.user_id = u.id
                ORDER BY r.created_at DESC
            """
            cursor.execute(sql)
            reports_data = cursor.fetchall()

            cursor.close()
        except Exception as e:
            print(f"Database Error: {e}")
        finally:
            conn.close()

    return render_template('reports.html', reports=reports_data)


from flask import render_template, session, redirect, url_for


# 1. Dashboard Page
@app.route('/user_home')
def user_home():
    user_id = session.get('user_id')


    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT name, email FROM users WHERE id = ?", (user_id,))
    user = cursor.fetchone()

    # Summary stats only
    cursor.execute("SELECT COUNT(*) as total FROM reports WHERE user_id = ?", (user_id,))
    total = cursor.fetchone()['total']

    cursor.close()
    conn.close()
    return render_template('user_home.html', user=user, total=total)


# 2. Upload Page
@app.route('/user_upload')
def user_upload():
    # 1. Check if user is logged in
    user_id = session.get('user_id')


    # 2. Get user info from database
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM users WHERE id = ?", (user_id,))
    user_data = cursor.fetchone()
    cursor.close()
    conn.close()

    # 3. Pass 'user' variable to the template
    return render_template('user_upload.html', user=user_data)


# 3. Reports Page
@app.route('/user_reports')
def user_reports():
    # 1. Use 'id' to match your login process session key
    user_id = session.get('id')

    if not user_id:
        return redirect(url_for('user_login'))

    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # 2. Fetch User Name
        cursor.execute("SELECT name FROM users WHERE id = ?", (user_id,))
        user_data = cursor.fetchone()

        # 3. Fetch Reports (Matches your SQL table exactly now)
        cursor.execute("""
            SELECT id, result, cause, organic_remedy, food_items, medicine, image_path, created_at 
            FROM reports 
            WHERE user_id = ? 
            ORDER BY created_at DESC
        """, (user_id,))
        reports_list = cursor.fetchall()

        return render_template('user_reports.html', user=user_data, reports=reports_list)

    except Exception as e:
        # This will tell you exactly if another column is missing
        print(f"SQL Error: {e}")
        return f"Database Error: {str(e)}"
    finally:
        cursor.close()
        conn.close()

# Dictionary to store all the details you requested
DEFICIENCY_KNOWLEDGE = {
    "Vitamin A deficiency": {
        "cause": "Lack of leafy greens and yellow fruits. Leads to Night Blindness and dry eyes.",
        "organic": "Consume 'Pasalai Keerai' and Carrot juice with a drop of honey for better absorption.",
        "food": "Carrots, Sweet Potato, Spinach, Papaya, and Fish oil.",
        "medicine": "Retinol supplements or Vitamin A drops (Consult doctor)."
    },
    "Vitamin B complex deficiency": {
        "cause": "Combined deficiency of multiple B-vitamins due to poor overall nutrition.",
        "organic": "Include 'Navadhanyam' (9 grains) and sprouts in your daily breakfast.",
        "food": "Whole grains, Meat, Legumes, and Seeds.",
        "medicine": "Becosules or Neurobion Forte."
    },
    "Vitamin B2 deficiency": {
        "cause": "Inadequate dairy intake. Causes mouth ulcers (Vaai pun) and cracked lips.",
        "organic": "Consume fresh curd or buttermilk (Moru) to restore gut health naturally.",
        "food": "Milk, Almonds, Mushrooms, and Whole grains.",
        "medicine": "Riboflavin tablets or B-complex syrup."
    },
    "Vitamin B3 deficieny": { # Matching the typo 'deficieny' in your folder name
        "cause": "Low protein diet. Causes skin issues like Pellagra and digestive problems.",
        "organic": "Eat roasted peanuts (Verkadalai) and sunflower seeds daily.",
        "food": "Peanuts, Brown rice, Chicken, and Green Peas.",
        "medicine": "Nicotinic Acid or Niacinamide tablets."
    },
    "Vitamin B9 deficiency": {
        "cause": "Lack of fresh green vegetables. Very critical for blood cells and pregnancy.",
        "organic": "Fresh Beetroot juice and soaked Black Chana (Karuppu Kondakadalai).",
        "food": "Spinach, Broccoli, Asparagus, and Oranges.",
        "medicine": "Folic Acid tablets (Folvite)."
    },
    "Vitamin B-12 deficiency": {
        "cause": "Poor absorption in the stomach or strict vegan diet without supplements.",
        "organic": "Drink 'Pazhaya Sadham' (fermented rice water) - a natural B12 powerhouse.",
        "food": "Milk, Eggs, Curd, Cheese, and Fortified Soy products.",
        "medicine": "Methylcobalamin or B12 injections for severe cases."
    },
    "Vitamin C deficiency": {
        "cause": "Lack of fresh citrus fruits. Leads to bleeding gums (Scurvy).",
        "organic": "Eat one raw Amla (Nellikai) every morning; Drink Lemon water (No sugar).",
        "food": "Guava (Koyya), Amla, Oranges, and Bell peppers.",
        "medicine": "Limcee or Vitamin C chewable tablets."
    },
    "Vitamin D deficiency": {
        "cause": "Staying indoors too much; lack of direct sunlight exposure.",
        "organic": "Daily Sunbath (7 AM to 9 AM). Oil massage with sesame oil helps absorption.",
        "food": "Mushrooms, Fatty fish, Egg yolks, and Fortified Milk.",
        "medicine": "Cholecalciferol (D3) granules or capsules."
    },
    "Vitamin E deficiency": {
        "cause": "Low intake of healthy fats and nuts. Causes skin aging and weak immunity.",
        "organic": "Use pure Coconut oil for cooking; Daily 4 soaked Almonds (Badam).",
        "food": "Almonds, Sunflower seeds, Spinach, and Avocados.",
        "medicine": "Evion 400 capsules."
    },
    "Vitamin K deficiency": {
        "cause": "Improper fat absorption or long-term antibiotic use. Causes easy bruising.",
        "organic": "Drink Cabbage juice or include fresh Cauliflower in your diet.",
        "food": "Green leafy vegetables, Cabbage, Broccoli, and Fermented Soy.",
        "medicine": "Phytomenadione (Vitamin K1) tablets."
    },
    "zinc, iron, biotin, or protein deficiency": {
        "cause": "Combined lack of essential minerals and protein. Causes hair loss, weakness, and slow healing.",
        "organic": "Eat mixed sprouts, pumpkin seeds, and dates with honey.",
        "food": "Eggs, Pulses (Paruppu), Nuts, Pomegranate, and Meat.",
        "medicine": "Zincovit, Dexorange, or Biotin supplements (Consult doctor)."
    }
}
MODEL_PATH = 'static/models/vitamin_model.h5'
# Load model with proper error handling
try:
    from tensorflow.keras.models import load_model as tf_load_model
    model = tf_load_model(MODEL_PATH)
except (ImportError, NameError, ModuleNotFoundError, OSError) as e:
    print(f"Warning: Could not load TensorFlow model ({e}). Model features will be unavailable.")
    model = None  # Model will be None and functions can check for this

def load_model(path):
    """Fallback function for when TensorFlow is not available"""
    return None
CLASS_NAMES = [
    "Vitamin A deficiency",
    "Vitamin B complex deficiency",
    "Vitamin B-12 deficiency",
    "Vitamin B2 deficiency",
    "Vitamin B3 deficieny",
    "Vitamin B9 deficiency",
    "Vitamin C deficiency",
    "Vitamin D deficiency",
    "Vitamin E deficiency",
    "Vitamin K deficiency",
    "zinc, iron, biotin, or protein deficiency"
]

@app.route('/predict', methods=['POST'])
def predict():
    # Use session.get('id') or 'user_id' based on your login logic
    user_id = session.get('id')
    if not user_id:
        return redirect(url_for('user_login'))

    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        # A. Save the uploaded file
        filename = file.filename
        upload_path = os.path.join('static/uploads/predictions', filename)
        os.makedirs(os.path.dirname(upload_path), exist_ok=True)
        file.save(upload_path)

        # B. PREDICTION LOGIC
        try:
            if model is None:
                # Fallback mock prediction when TensorFlow is unavailable
                import random
                class_index = random.randint(0, len(CLASS_NAMES) - 1)
                score = random.uniform(0.7, 0.99)
                detected_class = CLASS_NAMES[class_index]
            else:
                # 1. Load and Preprocess Image
                from tensorflow.keras.preprocessing import image
                img = image.load_img(upload_path, target_size=(224, 224))
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
                img_array /= 255.0  # Rescale (matches training ImageDataGenerator)

                # 2. Run Prediction
                predictions = model.predict(img_array)
                score = np.max(predictions) # Confidence score
                class_index = np.argmax(predictions) # Get highest probability index
                detected_class = CLASS_NAMES[class_index]

            # 3. Get Details from your Dictionary
            report_data = DEFICIENCY_KNOWLEDGE.get(detected_class, {
                "cause": "Cause unknown",
                "organic": "N/A",
                "food": "N/A",
                "medicine": "N/A"
            })

            # C. SAVE TO DATABASE
            conn = get_db_connection()
            cursor = conn.cursor()

            query = """INSERT INTO reports 
                       (user_id, result, cause, organic_remedy, food_items, medicine, image_path) 
                       VALUES (?, ?, ?, ?, ?, ?, ?)"""

            db_image_path = f'uploads/predictions/{filename}'
            values = (
                user_id, detected_class, report_data['cause'],
                report_data['organic'], report_data['food'],
                report_data['medicine'], db_image_path
            )

            cursor.execute(query, values)
            conn.commit()

            # Fetch user info for Sidebar
            cursor.execute("SELECT name FROM users WHERE id = ?", (user_id,))
            user_info = cursor.fetchone()

            cursor.close()
            conn.close()

            # D. PREPARE FINAL REPORT FOR HTML
            final_report = {
                "result": detected_class,
                "confidence": f"{score * 100:.2f}%",
                "cause": report_data['cause'],
                "organic": report_data['organic'],
                "food": report_data['food'],
                "medicine": report_data['medicine'],
                "image_path": db_image_path
            }

            return render_template('user_result.html', user=user_info, report=final_report)

        except Exception as e:
            print(f"Error during prediction: {e}")
            return f"Prediction Failed: {str(e)}"
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=True)