from flask import Flask,request
from gender_predict import predict

app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello World!"

@app.route('/predict_gender', methods=['GET', 'POST']) 
def form_example():
    if request.method == 'POST':  
        text = request.form.get('text')
        return '''<h1>prediction: {}</h1>'''.format(predict(text))

    return '''<form method="POST">
                  Text : <input type="text" name="text"><br>
                  <input type="submit" value="Submit"><br>
              </form>'''




if __name__ == "__main__":
    app.run(host='0.0.0.0', port='5001', debug=True)