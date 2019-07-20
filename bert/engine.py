from prediction import predict, tokenizer, label_list, estimator
from flask import Flask
from flask_restful import Resource, Api, reqparse

app = Flask(__name__)
api = Api(app)

class EntailmentEngine(Resource):
    @app.route('/predict', methods=['GET'])
    def get(self):
        parser = reqparse.RequestParser()
        parser.add_argument('text1', type=str, required=True, help='text1 cannot be empty')
        parser.add_argument('text2', type=str, required=True, help='text2 cannot be empty')
        args = parser.parse_args()
        return predict(args['text1'], args['text2'])

api.add_resource(EntailmentEngine, '/')

if __name__ == '__main__':
    app.run(debug=True)