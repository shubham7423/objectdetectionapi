import os
from flask import Flask, jsonify, request
from flask_restful import Api, Resource
from python1 import object_detection
import cv2
from flask import Flask, request, Response, jsonify, send_from_directory, abort


app = Flask(__name__)
# api = Api(app)

from python1 import object_detection

od = object_detection()
od.load_model()
# class Detections(Resource):
#     @staticmethod

@app.route('/')
def home():    
     return "Welcome"

@app.route('/detection', methods=['POST'])
def get_image():
    image = request.files["images"]
    image_name = image.filename
    image.save(os.path.join(os.getcwd(), image_name))
    image = cv2.imread(image_name)
    layerOutputs = od.detect_frame(image)
    image = od.draw_box(layerOutputs, image)
    # cv2.imwrite('detection.jpg', image)

    _, img_encoded = cv2.imencode('.png', image)
    response = img_encoded.tostring()

    os.remove(image_name)

    try:
        return Response(response=response, status=200, mimetype='image/png')
    except FileNotFoundError:
        abort(404)

# api.add_resource(Detections, '/predict', methods=['POST'])

if __name__ == '__main__':
    app.run(debug=True)
