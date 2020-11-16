from flask import Flask, render_template, url_for, request
import os
import numpy as np
import pandas as pd
import pickle

image_folder = os.path.join('static', 'images')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = image_folder

@app.route('/', methods=['GET', 'POST'])
def home():

    circle = os.path.join(app.config['UPLOAD_FOLDER'], 'Alaska circle.png')
    return render_template("Alaska.html", circle=circle)


@app.route('/reviews', methods=['POST', 'GET'])
def get_topic():

    circle = os.path.join(app.config['UPLOAD_FOLDER'], 'Alaska circle.png')
    df_pickle = os.path.join(app.config['UPLOAD_FOLDER'], 'for_app_df.pickle')

    if request.method == 'POST':
        topic_choice = request.form['topic-form']

        df = pickle.load(open(df_pickle, "rb"))
        df_1 = df[(df[f'Topic {topic_choice.split(" ")[1]} label'] == 1)]
        df_2 = df_1.sort_values(by=[topic_choice, "Sentiment net"], ascending=False)

        review = df_2.iloc[0]["Full review"]
        link = df_2.iloc[0]["Review link"]
        reviewer = df_2.iloc[0]["Reviewer"]
        property_name = df_2.iloc[0]["Property name"]
        when = df_2.iloc[0]["Date of stay"]
    
        
    return render_template("alaska2.html", circle=circle, review=review, link=link, reviewer=reviewer, property_name=property_name, when=when)


if __name__ == '__main__':
    app.run()







