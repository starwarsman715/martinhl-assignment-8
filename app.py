from flask import Flask, render_template, request, jsonify, send_from_directory
import os
from logistic_regression import do_experiments

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run_experiment', methods=['POST'])
def run_experiment():
    try:
        # Extract parameters from request
        data = request.get_json()
        start = float(data.get('start', 0.25))
        end = float(data.get('end', 2.0))
        step_num = int(data.get('step_num', 8))

        # Run experiments
        result = do_experiments(start, end, step_num)
        
        if result["status"] == "success":
            return jsonify({
                "status": "success",
                "message": result["message"],
                "dataset_img": result["dataset_img"],
                "parameters_img": result["parameters_img"]
            })
        else:
            return jsonify({
                "status": "error",
                "message": result["message"]
            }), 400

    except ValueError as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"An unexpected error occurred: {str(e)}"
        }), 500

@app.route('/results/<filename>')
def results(filename):
    try:
        return send_from_directory('results', filename)
    except FileNotFoundError:
        return jsonify({
            "status": "error",
            "message": f"File {filename} not found"
        }), 404

if __name__ == '__main__':
    app.run(debug=True)