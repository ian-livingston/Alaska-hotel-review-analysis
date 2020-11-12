from flask import Flask, render_template, url_for, request
import os

image_folder = os.path.join('static', 'images')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = image_folder

@app.route('/', methods=['GET', 'POST'])
def home():

    map1 = os.path.join(app.config['UPLOAD_FOLDER'], 'Alaska map.png')
    map2 = os.path.join(app.config['UPLOAD_FOLDER'], 'Alaska map 2.png')
    return render_template("Alaska.html", map_1 = map1, map_2 = map2)

if __name__ == '__main__':
    app.run()