from flask import Flask
from flask import request
from flask import render_template
from flask import abort
import model

app = Flask(__name__)
stored_text = ""

@app.route('/')
def index():
    global stored_text
    return render_template('main.html', stored_text=stored_text, uploaded_image="")

@app.route('/upload', methods=['POST'])
def upload():
    if (request.json['text'] == '' or request.json['image'] == ''):
        abort(400)
    global stored_text
    data = request.json
    image = data['image']
    stored_text = model.predict(image, "model.pt", data['text'])
    return {"image": data['image'], "text": stored_text}

if __name__ == '__main__':
    app.run(debug=True)

