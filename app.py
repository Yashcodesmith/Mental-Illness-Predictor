import numpy as np
from flask import Flask,request,render_template
import pickle

# Create Flask app
application = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

@application.route('/')
def pred():
    return render_template('pred.html')

@application.route("/prediction", methods=["POST"])
def prediction():
    # Get form data and convert to float
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    
    # Make prediction
    prediction = model.predict(final_features) 

    # Prepare the prediction message
    if prediction[0] == 1:
        prediction_text = "This person requires mental health treatment"
    else:
        prediction_text = "This person doesn't require mental health treatment"

    return render_template('pred.html', prediction_text=prediction_text)

if __name__ == "__main__":
    application.run(debug=True)
