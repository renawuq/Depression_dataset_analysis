import os
from flask import Flask, request, jsonify, render_template, session, redirect, url_for
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import plotly.express as px
import plotly
import json
from flask import Flask, render_template, jsonify
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression



app = Flask(__name__)
app.secret_key = os.urandom(24)

try:
    df = pd.read_csv('website/cleans.csv')
except FileNotFoundError:
    print("Error: cleans.csv file not found. Ensure the file path is correct.")
    df = pd.DataFrame(columns=['Gender', 'Age', 'Work Pressure','Job Satisfaction','Sleep Duration','Dietary Habits', 'Have you ever had suicidal thoughts ?','Work Hours','Financial Stress','Family History of Mental Illness','Depression']) 
# Add a row number column to your DataFrame
df.insert(0, 'Row Number', range(1, len(df) + 1))





def create_boxplot(columns):
    data = [df[column] for column in columns]
    # plt.figure(figsize=(20, 10))
    plt.boxplot(data, labels=columns)
    plt.title('Boxplots of Some of the factors')
    plt.ylabel('Values')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    image_io = io.BytesIO()
    plt.savefig(image_io, format='png')
    plt.close()
    image_io.seek(0)
    image_data = base64.b64encode(image_io.read()).decode('utf-8')
    return image_data

def get_boxplot_and_outliers(columns):
    if df.empty:
        return "No data available for plotting.", "No data available for analysis."

    mappings = {
        "Sleep Duration": {
            0: 'less than 5 hours',
            1: '5-6 hours',
            2: '7-8 hours',
            3: 'more than 8 hours'
        },
        "Dietary Habits": {
            0: 'unhealthy',
            1: 'moderate',
            2: 'healthy'
        },
        "Gender": {
            0: 'male',
            1: 'female'
        }
    }
    image_data = create_boxplot(columns)
    final_info = ""

    for column in columns:
        if column not in df:
            final_info += f"Column {column} not found in the dataset.\n"
            continue

        temp = df[column].mode().iloc[0] if not df[column].dropna().empty else "Unknown"
        temp = mappings.get(column, {}).get(temp, temp)
        most_common_count = df[column].value_counts().iloc[0] if not df[column].dropna().empty else 0
        final_info += f"Most common value in {column}: {temp} with {most_common_count} occurrences.\n\n"

    return f'<img src="data:image/png;base64,{image_data}">', final_info


# @app.route('/')
# def index():
#     return render_template('index.html')

@app.route('/', methods=['GET', 'POST'])
def index():
    if 'selected_columns' not in session:
        session['selected_columns'] = []

    if request.method == 'POST':
        action = request.form.get('action')
        if action == 'Add Columns':
            selected_columns = session['selected_columns']
            selected_columns += request.form.getlist('columns')
            session['selected_columns'] = list(set(selected_columns)) 
        elif action == 'Delete All':
            session['selected_columns'] = []

        return redirect(url_for('index'))

    boxplot_output = ""
    outliers_output = ""
    selected_columns = session['selected_columns']
    if selected_columns:
        boxplot_output, outliers_output = get_boxplot_and_outliers(selected_columns)

    return render_template('index.html', selected_columns=selected_columns, boxplot_output=boxplot_output, outliers_output=outliers_output)

@app.route('/analyze')
def analyze():
    return render_template('analyze.html')

@app.route('/pca_data')
def pca_data():
    try:
        df = pd.read_csv('website/cleans.csv')
    except FileNotFoundError:
        print("Error: cleans.csv file not found. Ensure the file path is correct.")
        df = pd.DataFrame(columns=['Gender', 'Age', 'Work Pressure','Job Satisfaction','Sleep Duration','Dietary Habits', 'Have you ever had suicidal thoughts ?','Work Hours','Financial Stress','Family History of Mental Illness','Depression']) 
    features = df.drop(columns=['Depression'])
    target = df['Depression']

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(features_scaled)

    pca_df = pd.DataFrame(data=principal_components, columns=['Principal Component 1', 'Principal Component 2'])
    pca_df['Depression'] = target.map({1: 'Depression', 0: 'No Depression'})
    pca_df['Age'] = df['Age']
    pca_df['Gender'] = df['Gender'].map({1: 'Female', 0: 'Male'})

    pca_result = {
        "x": pca_df['Principal Component 1'].tolist(),
        "y": pca_df['Principal Component 2'].tolist(),
        "text": pca_df.apply(lambda row: f"Age: {row['Age']}, Gender: {row['Gender']}", axis=1).tolist(),
        "color": pca_df['Depression'].tolist()
    }
    return jsonify(pca_result)


@app.route("/api/person/<int:person_index>")
def get_person(person_index):
    mappings = {
        "Sleep Duration": {
            0: 'less than 5 hours',
            1: '5-6 hours',
            2: '7-8 hours',
            3: 'more than 8 hours'
        },
        "Dietary Habits": {
            0: 'unhealthy',
            1: 'moderate',
            2: 'healthy'
        },
        "Gender": {
            0: 'male',
            1: 'female'
        }
    }
    if person_index < 0 or person_index >= len(df):
        return jsonify({"error": "Person not found"}), 404
    person = df.iloc[person_index].to_dict()
    mapped_data = {
        "gender": mappings["Gender"].get(person["Gender"], "unknown"),
        "age": person["Age"],
        "workPressure": person["Work Pressure"],
        "jobSatisfaction": person["Job Satisfaction"],
        "sleepDuration": mappings["Sleep Duration"].get(person["Sleep Duration"], "unknown"),
        "dietaryHabits": mappings["Dietary Habits"].get(person["Dietary Habits"], "unknown"),
        "suicidalThoughts": "Yes" if person["Have you ever had suicidal thoughts ?"] == 1 else "No",
        "workHours": person["Work Hours"],
        "financialStress": person["Financial Stress"],
        "familyHistory": "Yes" if person["Family History of Mental Illness"] == 1 else "No",
        "depression": "Yes" if person["Depression"] == 1 else "No"
    }
    return jsonify(mapped_data)

@app.route('/barplot')
def barplot():
    try:
        df = pd.read_csv('website/cleans.csv')    
    except FileNotFoundError:
        print("Error: cleans.csv file not found. Ensure the file path is correct.")
        df = pd.DataFrame(columns=['Gender', 'Age', 'Work Pressure','Job Satisfaction','Sleep Duration','Dietary Habits', 'Have you ever had suicidal thoughts ?','Work Hours','Financial Stress','Family History of Mental Illness','Depression']) 
    df["DepressionNumeric"] = df["Depression"].map({"Yes": 1, "No": 0})
    df["Gender"] = df["Gender"].map({1: "Male", 0: "Female"})
    df["Have you ever had suicidal thoughts ?"] = df["Have you ever had suicidal thoughts ?"].map({1: "Yes", 0: "No"})
    df["Depression"] = df["Depression"].map({1: "Yes", 0: "No"})

    depression_counts = df["Depression"].value_counts()
    return render_template("barplot.html",
                           depression_labels=depression_counts.index.tolist(),                           
                           depression_values=depression_counts.values.tolist())

@app.route('/KNN_model', methods=['GET', 'POST'])
def KNN_model():
    prediction = None
    accuracy_data = {'labels': [], 'values': []}
    image_path = 'static/healthcare.png'  
    try:
        df = pd.read_csv('website/cleans.csv')
    except FileNotFoundError:
        print("Error: cleans.csv file not found.")
        df = pd.DataFrame(columns=['Gender', 'Age', 'Work Pressure', 'Job Satisfaction',
                                   'Sleep Duration', 'Dietary Habits', 'Have you ever had suicidal thoughts ?',
                                   'Work Hours', 'Financial Stress', 'Family History of Mental Illness', 'Depression'])

    X = df[['Gender', 'Age', 'Work Pressure', 'Job Satisfaction', 'Sleep Duration', 
            'Dietary Habits', 'Have you ever had suicidal thoughts ?', 
            'Work Hours', 'Financial Stress', 'Family History of Mental Illness']]
    y = df['Depression']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)

    smote = SMOTE(random_state=2)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_test_scaled = scaler.transform(X_test)

    
    if request.method == 'POST':
        # Collect user input from form
        gender = request.form['gender']
        age = int(request.form['age'])
        work_pressure = int(request.form['work_pressure'])
        job_satisfaction = int(request.form['job_satisfaction'])
        sleep_duration = request.form['sleep_duration']
        dietary_habits = request.form['dietary_habits']
        suicidal_thoughts = request.form['suicidal_thoughts']
        work_hours = int(request.form['work_hours'])
        financial_stress = int(request.form['financial_stress'])
        family_history = request.form['family_history']
        k_neighbors = int(request.form['k_neighbors'])

        # Convert categorical inputs to numbers using label encoding
        gender = 0 if gender == "Male" else 1

        if sleep_duration == "<5 hours":
            sleep_duration = 0
        elif sleep_duration == "5-6 hours":
            sleep_duration = 1
        elif sleep_duration == "7-8 hours":
            sleep_duration = 2
        else:  # >8 hours
            sleep_duration = 3

        # Dietary Habits encoding: Unhealthy = 0, Moderate = 1, Healthy = 2
        if dietary_habits == "Unhealthy":
            dietary_habits = 0
        elif dietary_habits == "Moderate":
            dietary_habits = 1
        else:  # Healthy
            dietary_habits = 2

        # Suicidal Thoughts encoding: Yes = 1, No = 0
        suicidal_thoughts = 1 if suicidal_thoughts == "Yes" else 0

        # Family History encoding: Yes = 1, No = 0
        family_history = 1 if family_history == "Yes" else 0
        
        # Prepare input features
        input_data = [[gender, age, work_pressure, job_satisfaction, sleep_duration, dietary_habits,
                       suicidal_thoughts, work_hours, financial_stress, family_history]]

        # Scale the input data using the same scaler used for training
        input_data_scaled = scaler.transform(input_data)

        knn = KNeighborsClassifier(n_neighbors=k_neighbors)
        knn.fit(X_train_scaled, y_train_resampled)

        # Calculate accuracy for visualization
        train_accuracy = knn.score(X_train_scaled, y_train_resampled)
        test_accuracy = knn.score(X_test_scaled, y_test)

        accuracy_data = {
            'labels': ['Training Accuracy', 'Testing Accuracy'],
            'values': [train_accuracy * 100, test_accuracy * 100]
        }

        # Predict depression
        prediction = knn.predict(input_data_scaled)[0]
        prediction = 'Depressed' if prediction == 1 else 'Not Depressed'

        if prediction == 'Depressed':
            image_path = 'image/emo1.jpg' 
        else:
            image_path = 'image/happy1.jpg'  
    return render_template('KNN_model.html', image_path=image_path, prediction=prediction, accuracy_data=accuracy_data)

@app.route('/LR_model', methods=['GET', 'POST'])
def LR_model():
    prediction = None
    accuracy_data = {'labels': [], 'values': []}
    image_path = 'static/healthcare.png'  
    try:
        df = pd.read_csv('website/cleans.csv')
    except FileNotFoundError:
        print("Error: cleans.csv file not found.")
        df = pd.DataFrame(columns=['Gender', 'Age', 'Work Pressure', 'Job Satisfaction',
                                   'Sleep Duration', 'Dietary Habits', 'Have you ever had suicidal thoughts ?',
                                   'Work Hours', 'Financial Stress', 'Family History of Mental Illness', 'Depression'])

    X = df[['Gender', 'Age', 'Work Pressure', 'Job Satisfaction', 'Sleep Duration', 
            'Dietary Habits', 'Have you ever had suicidal thoughts ?', 
            'Work Hours', 'Financial Stress', 'Family History of Mental Illness']]
    y = df['Depression']
    

    
    if request.method == 'POST':
        # Collect user input from form
        gender = request.form['gender']
        age = int(request.form['age'])
        work_pressure = int(request.form['work_pressure'])
        job_satisfaction = int(request.form['job_satisfaction'])
        sleep_duration = request.form['sleep_duration']
        dietary_habits = request.form['dietary_habits']
        suicidal_thoughts = request.form['suicidal_thoughts']
        work_hours = int(request.form['work_hours'])
        financial_stress = int(request.form['financial_stress'])
        family_history = request.form['family_history']
        training = 1- float(request.form['training'])
        print(training)

        # Convert categorical inputs to numbers using label encoding
        gender = 0 if gender == "Male" else 1

        if sleep_duration == "<5 hours":
            sleep_duration = 0
        elif sleep_duration == "5-6 hours":
            sleep_duration = 1
        elif sleep_duration == "7-8 hours":
            sleep_duration = 2
        else:  # >8 hours
            sleep_duration = 3

        # Dietary Habits encoding: Unhealthy = 0, Moderate = 1, Healthy = 2
        if dietary_habits == "Unhealthy":
            dietary_habits = 0
        elif dietary_habits == "Moderate":
            dietary_habits = 1
        else:  # Healthy
            dietary_habits = 2

        # Suicidal Thoughts encoding: Yes = 1, No = 0
        suicidal_thoughts = 1 if suicidal_thoughts == "Yes" else 0

        # Family History encoding: Yes = 1, No = 0
        family_history = 1 if family_history == "Yes" else 0
        
        # Prepare input features
        input_data = [[gender, age, work_pressure, job_satisfaction, sleep_duration, dietary_habits,
                       suicidal_thoughts, work_hours, financial_stress, family_history]]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=training, random_state=2)

        smote = SMOTE(random_state=2)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_resampled)
        X_test_scaled = scaler.transform(X_test)        
        input_data_scaled = scaler.transform(input_data)

        model = LogisticRegression()
        model.fit(X_train_scaled, y_train_resampled)

        # Calculate accuracy for visualization
        train_accuracy = model.score(X_train_scaled, y_train_resampled)
        test_accuracy = model.score(X_test_scaled, y_test)

        accuracy_data = {
            'labels': ['Training Accuracy', 'Testing Accuracy'],
            'values': [train_accuracy * 100, test_accuracy * 100]
        }

        # Predict depression
        prediction = model.predict(input_data_scaled)[0]
        prediction = 'Depressed' if prediction == 1 else 'Not Depressed'

        if prediction == 'Depressed':
            image_path = 'image/emo.jpeg' 
        else:
            image_path = 'image/happy.jpg'  
    return render_template('LR_model.html', image_path=image_path, prediction=prediction, accuracy_data=accuracy_data)



if __name__ == "__main__":
    app.run(debug=True)
