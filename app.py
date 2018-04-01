from flask import Flask, render_template, request, json
from haiku import combiner

app = Flask(__name__) # create the application instance
combiner = combiner.Combiner()

@app.route("/new_haiku", methods=['GET'])
def new_haiku():
    im, haiku = combiner.make_new_haiku()
    path = "static/img/haiku.png"
    im.save(path)
    return json.dumps({'path': path, 'haiku': haiku}), 200, {'Content-Type':'application/json'}

@app.route("/")
def home():
    return render_template('layout.html')

if __name__ == '__main__':
    app.run(debug=True)
