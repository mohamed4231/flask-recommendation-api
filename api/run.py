from flask import Flask, request, jsonify
from the_recommender import Recommender

recommender = Recommender()
recommender.load_model()
app = Flask(__name__)


@app.route("/recommend", methods=['POST'])
def get_employee_record():
    customer_info = request.get_json()
    products = recommender.recommend(customer_info)
    return jsonify(products)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
