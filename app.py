from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load model and column names
model = pickle.load(open('student_performance_model.pkl', 'rb'))
model_columns = pickle.load(open('model_columns.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Collect form inputs
    gender = request.form['gender']
    race = request.form['race']
    parent_edu = request.form['parent_edu']
    lunch = request.form['lunch']
    test_prep = request.form['test_prep']
    math_score = float(request.form['math_score'])
    reading_score = float(request.form['reading_score'])
    writing_score = float(request.form['writing_score'])

    # Build input dictionary for encoding
    data = {
        'math_score': math_score,
        'reading_score': reading_score,
        'writing_score': writing_score,
    }

    # One-hot encoding manually (same as during training)
    data[f'gender_{gender}'] = 1
    data[f'race/ethnicity_{race}'] = 1
    data[f'parental_level_of_education_{parent_edu}'] = 1
    data[f'lunch_{lunch}'] = 1
    data[f'test_preparation_course_{test_prep}'] = 1

    # Convert to DataFrame and align columns
    input_df = pd.DataFrame([data])
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    # Predict
    prediction = model.predict(input_df)[0]

    return render_template('index.html', prediction_text=f"Predicted Score: {round(prediction, 2)}")

if __name__ == '__main__':
    app.run(debug=True)
