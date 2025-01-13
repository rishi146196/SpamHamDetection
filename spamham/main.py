from flask import Flask, request, render_template
import pickle

# Load the trained model and vectorizer
with open('vectorizer.pkl', 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)

with open('spam_ham_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the email subject from the form
    email_subject = request.form['email_subject']
    
    # Preprocess and predict
    email_vec = vectorizer.transform([email_subject])
    prediction = model.predict(email_vec)[0]
    
    # Interpret the result
    result = "Spam" if prediction == 1 else "Ham"
    
    return render_template('index.html', prediction=f"The email is classified as: {result}")

if __name__ == '__main__':
    app.run(debug=True)
