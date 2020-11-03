import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

import pprint

class LoggingMiddleware(object):
    def __init__(self, app):
        self._app = app

    def __call__(self, env, resp):
        errorlog = env['wsgi.errors']
        pprint.pprint(('REQUEST', env), stream=errorlog)

        def log_response(status, headers, *args):
            pprint.pprint(('RESPONSE', status, headers), stream=errorlog)
            return resp(status, headers, *args)

        return self._app(env, log_response)


app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''

    req=str(request.__dict__)
    print(request.form)
    int_features = [str(x) for x in request.form.values()]
    print(int_features)
    # final_features = [np.array(int_features)]
    # prediction = model.predict(final_features)

    # output = round(prediction[0], 2)
    if('nigga' in req):
        return render_template('index.html', prediction_text='The request is malicious')
    else:
        return render_template('index.html', prediction_text='The request is not malicious')


@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)