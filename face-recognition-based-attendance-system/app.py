import sqlite3
import cv2
import os
import shutil  # Import shutil module
import mysql.connector  # Import MySQL connector
from flask import Flask, request, render_template, redirect, session, url_for, flash, send_file, make_response
import io
from io import BytesIO
from datetime import date, datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from werkzeug.security import generate_password_hash
from werkzeug.security import check_password_hash
import pandas as pd
import joblib
import csv

# Flask App Initialization
app = Flask(__name__)
app.secret_key = os.urandom(24) 

# Date today in two formats
datetoday = date.today().strftime("%Y-%m-%d")
datetoday2 = date.today().strftime("%d-%B-%Y")

# MySQL Database Configuration
def get_db_connection():
    conn = mysql.connector.connect(
        host='localhost',
        user='root',
        password='',
        database='attendance_system'
    )
    return conn

# Initialize Face Detector
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Create necessary directories
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')

# Ensure attendance file exists
attendance_file = 'Attendance/attendance_records.csv'
if not os.path.isfile(attendance_file):
    with open(attendance_file, 'w') as f:
        f.write('Name,Roll,Teacher,Subject,Subject_ID,Date,Time\n')


# Get total registered users
def totalreg():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM users")
    total = cursor.fetchone()[0]
    conn.close()
    return total

# Extract faces from an image
def extract_faces(frame):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
    return faces

# Identify face using ML model
def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)

# Train model on registered faces
def train_model():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT name, roll FROM users")
    users = cursor.fetchall()
    conn.close()

    faces = []
    labels = []
    
    for user in users:
        user_folder = f'static/faces/{user[0]}_{user[1]}'
        if os.path.isdir(user_folder):
            for imgname in os.listdir(user_folder):
                img_path = f'{user_folder}/{imgname}'
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
                resized_face = cv2.resize(img, (50, 50)).flatten()  # Flatten image for training
                faces.append(resized_face)
                labels.append(f'{user[0]}_{user[1]}')

    if len(faces) > 0:
        faces = np.array(faces)
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(faces, labels)
        joblib.dump(knn, 'static/face_recognition_model.pkl')
        print("Model trained successfully.")
    else:
        print("No images found for training.")

# Add attendance to the CSV file
def add_attendance(name, subject, subject_id):
    username, userid = name.split('_')
    current_time = datetime.now().strftime("%H:%M:%S")
    with open(attendance_file, 'a') as f:
        f.write(f'{username},{userid},{subject},{subject_id},{datetoday},{current_time}\n')

def getallusers():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT name, roll FROM users")
    users = cursor.fetchall()
    conn.close()
    return users, [user[0] for user in users], [user[1] for user in users], len(users)

@app.route('/download_attendance')
def download_attendance():
    file_path = "..\\Attendance\\attendance_records.csv"
    return send_file(file_path, as_attachment=True)

def get_all_teachers():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM Teachers")
    teachers = cursor.fetchall()
    conn.close()
    return teachers

def del_user(roll_number):
    conn = get_db_connection()
    cursor = conn.cursor()

    # 1. Fetch the username before deleting the user
    cursor.execute("SELECT name FROM users WHERE roll = %s", (roll_number,))
    user = cursor.fetchone()  # Fetch one record

    if user:
        username = user[0]  # Extract the username
    else:
        print(f"No user found with roll number {roll_number}")
        conn.close()
        return  # Exit function if user doesn't exist

    # 2. Delete the user from the database
    sql = "DELETE FROM users WHERE roll = %s"
    values = (roll_number,)  # Ensure it's a tuple
    cursor.execute(sql, values)
    conn.commit()

    # 3. Construct the folder path
    folder_path = os.path.join("static", "faces", f"{username}_{roll_number}")

    # 4. Delete the folder if it exists
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)  # Deletes the entire folder and contents
        print(f"Folder '{folder_path}' deleted successfully")
    else:
        print(f"Folder '{folder_path}' does not exist")

    # 5. Close DB connection
    conn.close()


# Routing Functions
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/admin')
def admin_page():
    return render_template('admin.html')

@app.route('/logout', methods=['POST'])
def logout():
    session.clear()
    return redirect(url_for('home'))

@app.route('/register_teacher', methods=['GET', 'POST'])
def register_teacher():
    if request.method == 'POST':
        teacher_name = request.form['teacher_name']
        teacher_email = request.form['teacher_email']
        teacher_phone = request.form.get('teacher_phone', '')
        teacher_department = request.form.get('teacher_department', '')
        teacher_password = request.form['teacher_password']  # Get password from form

        # Hash the password before storing
        hashed_password = generate_password_hash(teacher_password)

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM Teachers WHERE email = %s", (teacher_email,))
        existing_teacher = cursor.fetchone()
        
        if existing_teacher:
            flash(message="This email is already registered.", category="error")
            conn.close()
            return render_template('registration_teacher.html')

        # Insert teacher into the database
        cursor.execute("""
            INSERT INTO Teachers (name, email, phone_number, department, password)
            VALUES (%s, %s, %s, %s, %s)
        """, (teacher_name, teacher_email, teacher_phone, teacher_department, hashed_password))
        conn.commit()
        conn.close()

        flash("Teacher successfully registered!", "success")
        return redirect(url_for('register_teacher'))

    return render_template('registration_teacher.html')
  # This will render the teacher registration page

@app.route('/register_subject', methods=['GET', 'POST'])
def register_subject():
    
    if request.method == 'POST':
        subject_name = request.form['subject_name']
        subject_code = request.form['subject_code']
        teacher_id = int(request.form['teacher_id'])
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM Subjects WHERE code = %s", (subject_code,))
        existing_teacher = cursor.fetchone()
        
        if existing_teacher:
            # Email already exists
            flash(message="This subject is already registered.", category="error")
            cursor.execute("SELECT teacher_id, name FROM Teachers")
            teachers = cursor.fetchall()  # Fetch all teachers to display in a dropdown

            conn.close()

            return render_template('register_subject.html', teachers=teachers) 
        # Insert subject into the database
        
        cursor.execute("""
            INSERT INTO Subjects (name, code, teacher_id)
            VALUES (%s, %s, %s)
        """, (subject_name, subject_code, teacher_id))
        conn.commit()
        conn.close()

        flash("Subject successfully registered!", "success")
        return redirect(url_for('register_subject'))
    conn = get_db_connection()
    cursor = conn.cursor()

    # Example query to fetch subjects or other necessary data
    cursor.execute("SELECT teacher_id, name FROM Teachers")
    teachers = cursor.fetchall()  # Fetch all teachers to display in a dropdown

    conn.close()

    return render_template('register_subject.html', teachers=teachers) 

@app.route('/teacher_login', methods=['GET', 'POST'])
def teacher_login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM Teachers WHERE email = %s", (email,))
        teacher = cursor.fetchone()
        conn.close()

        if teacher and check_password_hash(teacher['password'], password):
            # Login successful
            session['teacher_id'] = teacher['teacher_id']
            session['teacher_name'] = teacher['name']
            session['email'] = teacher['email']  # ✅ Required for dashboard
            flash("Login successful!", "success")
            return redirect(url_for('teacher_dashboard'))
        else:
            flash("Invalid email or password.", "error")
            return render_template('teacher_login.html')

    return render_template('teacher_login.html')

@app.route('/teacher_dashboard', methods=['GET', 'POST'])
def teacher_dashboard():
    if 'teacher_id' not in session:
        return redirect(url_for('teacher_login'))

    teacher_id = session['teacher_id']

    # Get teacher info
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM Teachers WHERE teacher_id = %s", (teacher_id,))
    teacher = cursor.fetchone()

    # Get all subjects assigned to this teacher
    cursor.execute("SELECT name FROM Subjects WHERE teacher_id = %s", (teacher_id,))
    subjects = cursor.fetchall()
    subject_names = [sub['name'] for sub in subjects]
    conn.close()

    # Get selected subject from form (if any)
    selected_subject = request.form.get('subject_filter') if request.method == 'POST' else None

    # Read attendance from CSV
    attendance_data = []
    attendance_file = 'Attendance/attendance_records.csv'
    if os.path.isfile(attendance_file):
        with open(attendance_file, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row['Subject'] in subject_names:
                    if not selected_subject or row['Subject'] == selected_subject:
                        attendance_data.append(row)

    return render_template('teacher_dashboard.html',
                           teacher=teacher,
                           subjects=subject_names,
                           selected_subject=selected_subject,
                           attendance=attendance_data)

@app.route('/student', methods=['GET', 'POST'])
def student_page():
    if request.method == 'POST':
        username = request.form.get('username')
        roll_no = int(request.form.get('roll_no'))  # Ensure this matches the form field
        
        # Connect to the database
        conn = get_db_connection()
        cursor = conn.cursor()

        try:
            # Execute the SQL query to check if the user exists
            cursor.execute("SELECT name, roll FROM users WHERE name = %s AND roll = %s", (username, roll_no))
            user = cursor.fetchone()

            if user:
                # If user is found, store user info in session and redirect to the dashboard
                session['username'] = user[0]
                session['roll'] = user[1]
                return redirect(url_for('show_student_dashboard'))
            else:
                # If username and roll do not match, show error message
                return render_template('student.html', message='Invalid username or roll number')

        except mysql.connector.Error as err:
            # In case of an error in the database operation
            return render_template('student.html', message=f"Database error: {err}")
        
    return render_template('student.html')

@app.route('/student_dashboard')
def show_student_dashboard():
    # If no user is logged in, redirect to the login page
    if 'username' not in session:
        return redirect(url_for('student_page'))
    
    # Get student details from session
    username = session['username']
    roll = session['roll']

    attendance_file = 'Attendance/attendance_records.csv'
    
    try:
        df = pd.read_csv(attendance_file)
        
        # Debugging: Print CSV column names to check for mismatches
        print("CSV Columns:", df.columns)

        # Ensure correct column names (remove spaces if necessary)
        df.columns = df.columns.str.strip()

        # Filter data by student roll number (convert roll to int if needed)
        df_filtered = df[df['Roll'] == int(roll)]

        # Convert DataFrame to a list of dictionaries for template rendering
        attendance_records = df_filtered.to_dict(orient='records')

    except Exception as e:
        print(f"Error reading attendance file: {e}")
        attendance_records = []

    # Render the student dashboard template correctly
    return render_template('studentdahsboard.html', username=username, roll=roll, attendance_records=attendance_records)

@app.route('/download_student_attendance')
def download_student_attendance():
    if 'username' not in session:
        return redirect(url_for('student_page'))  # Redirect if not logged in
    
    # Get student details from session
    username = session['username']
    roll = str(session['roll'])  # Convert roll to string for filtering

    attendance_file = 'Attendance/attendance_records.csv'

    try:
        df = pd.read_csv(attendance_file, dtype={'Roll': str})  # Read Roll column as string
        
        # Ensure column names are stripped of spaces
        df.columns = df.columns.str.strip()
        
        # Filter attendance data by student roll number
        df_filtered = df[df['Roll'].str.strip() == roll]  # Compare as string

        if df_filtered.empty:
            return "No attendance records found for this student."

        # Convert DataFrame to CSV in-memory (without saving to disk)
        output = io.BytesIO()
        df_filtered.to_csv(output, index=False, encoding='utf-8')  # Write CSV to memory
        output.seek(0)  # Move cursor to the beginning

        # Send the file as an attachment (without saving it)
        return send_file(output, mimetype='text/csv', as_attachment=True, download_name=f"{username}_attendance.csv")

    except Exception as e:
        return f"Error processing the file: {e}"
    
@app.route('/listusers', methods = ['GET', 'POST'])
def listusers():
    if request.method == 'POST':
        roll_to_delete = request.form.get("roll_to_delete")
        if roll_to_delete:
            del_user(roll_to_delete)
        return redirect(url_for('listusers'))  # Redirect to refresh the page

    userlist, names, rolls, l = getallusers()
    return render_template('listusers.html', userlist=userlist, names=names, rolls=rolls, l=l, totalreg=totalreg(), datetoday2=datetoday2)

@app.route('/attendance', methods=['GET', 'POST'])
def attendance():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM Subjects")  # Ensure correct column names
    subjects = cursor.fetchall()  # Fetch all subjects
    conn.close()
    print(subjects)
    if request.method == "POST":
        username = request.form.get('username')
        password = request.form.get('password')
        if username == 'admin' and password == 'admin123':
            return render_template('attendance.html', subjects=subjects)
        else:
            return render_template('admin.html', mess='Invalid Credentials', subjects=subjects)
    return render_template('attendance.html', subjects=subjects)

capture_running = False  # Global variable to control capture loop

@app.route('/start', methods=['GET', 'POST'])
def start():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    if request.method == "POST":
        subject_id = int(request.form.get('subject_id'))
        cursor.execute(f"SELECT * FROM Subjects where subject_id= %s",(subject_id,) )  # Ensure correct column names
        subjects = cursor.fetchone()  # Fetch all subjects
        conn.close()

        
        subject_name = subjects[1]
        subject_id = subjects[0]  # Assuming subject_name is the second column
    
    global capture_running
    if 'face_recognition_model.pkl' not in os.listdir('static'):
        flash('No trained model found. Please add a new face first.')
        return render_template('attendance.html', totalreg=totalreg(), datetoday2=datetoday2)

    subject = request.args.get('subject', subject_name)
    subject_id = request.args.get('subject_id', subject_id)

    cap = cv2.VideoCapture(0)
    capture_running = True  
    detected_faces = set()  # Store already detected face IDs

    while capture_running:
        ret, frame = cap.read()
        if not ret:
            break

        faces = extract_faces(frame)  # Extract faces from frame
        for (x, y, w, h) in faces:
            face_img_gray = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
            face = cv2.resize(face_img_gray, (50, 50))
            identified_person = identify_face(face.reshape(1, -1))[0]  # Identify person

            # Check if this person has already been detected
            if identified_person not in detected_faces:
                detected_faces.add(identified_person)  # Add to detected set
                add_attendance(identified_person, subject, subject_id)  # Mark attendance

            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, identified_person, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Attendance', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return redirect(url_for('attendance'))

@app.route('/stop', methods=['GET'])
def stop():
    global capture_running
    capture_running = False  # Set flag to False to stop loop
    return redirect(url_for('attendance'))  # Redirect to attendance page

from flask import redirect, url_for, render_template
@app.route('/add', methods=['POST'])

def add_user():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    userimagefolder = f'static/faces/{newusername}_{newuserid}'

    if not os.path.exists(userimagefolder):
        os.makedirs(userimagefolder)

    cap = cv2.VideoCapture(0)
    i = 0

    while i < 10:  # Capture exactly 10 images
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image from camera.")
            break

        faces = extract_faces(frame)

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]

            # Ensure extracted face is not empty
            if face_img.size == 0 or w == 0 or h == 0:
                print("Invalid face detected, skipping frame.")
                continue

            face_img_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

            # Ensure grayscale image has valid shape before resizing
            if face_img_gray.shape[0] > 0 and face_img_gray.shape[1] > 0:
                face_img_resized = cv2.resize(face_img_gray, (50, 50))  # Resize to 50x50

                img_path = f'{userimagefolder}/{i}.jpg'
                cv2.imwrite(img_path, face_img_resized)
                i += 1
                print(f"Saved {img_path}")

            # Draw rectangle and display progress
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
            cv2.putText(frame, f'Images Captured: {i}/10', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)

            if i >= 10:  # Stop capturing after 10 images
                break

        cv2.imshow('Adding User', frame)
        if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
            break

    cap.release()
    cv2.destroyAllWindows()

    print("User added successfully. Training model now.")

    # Add user to database
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO users (name, roll) VALUES (%s, %s)", (newusername, newuserid))
    conn.commit()
    conn.close()

    # Call the function to train the model
    train_model()

    # Prepare a success message or redirect to a new page
    MESSAGE = 'User added successfully and model trained.'

    # You can either return a template or a redirect
    return render_template('home.html', mess=MESSAGE)

    # Or use a redirect to another route
    # return redirect(url_for('some_other_route'))


if __name__ == '__main__':
    app.run(debug=True, port=1000)