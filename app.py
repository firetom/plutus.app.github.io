from flask import Flask, render_template, request, redirect, url_for
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from tensorflow.keras.models import load_model

df = pd.read_csv('static/concrete_data.csv')
X = df.drop('concrete_compressive_strength', axis=1).values
scale = MinMaxScaler()
X = scale.fit_transform(X)
model = load_model('D:/Intro To Civil/static/final_model.h5')

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/pred', methods=['GET', 'POST'])
def pred():
    if request.method == "POST":
        a = float(request.form["a"])
        b = float(request.form["b"])
        c = float(request.form["c"])
        d = float(request.form["d"])
        e = float(request.form["e"])
        f = float(request.form["f"])
        g = float(request.form["g"])
        h = float(request.form["h"])
        l=[a,b,c,d,e,f,g,h]
        l=np.array(l)
        l=l.reshape(-1,8)
        scaled_l=scale.transform(l)
        x=model.predict(scaled_l)
        x=x[0][0]
        return redirect(url_for("p",usr=x))
    else:
        return render_template("home.html")


@app.route("/<usr>")
def p(usr):
    return render_template('test.html',usr=usr)

if __name__ == "__main__":
    app.run(port=7800)
