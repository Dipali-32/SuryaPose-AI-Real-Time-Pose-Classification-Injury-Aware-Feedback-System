

# from flask import Flask, request, jsonify
# import numpy as np
# import tensorflow as tf
# from flask_cors import CORS

# app = Flask(__name__)
# CORS(app)

# # Load the trained Keras model
# model = tf.keras.models.load_model("suryanamaskar2_by_deeps.h5")

# # Pose class labels
# class_labels = [
#     "Pranamasana",
#     "HastaUttasana",
#     "ashwa _anchalanasana",
#     "bhujangasana",
#     "kumbhakasana",
#     "ashtanga_namaskara",
#     "Padahastasana",
#     "adho_mukh_svanasana"
# ]

# # Threshold settings
# kp_threshold = 0.3      # minimum confidence score for a keypoint to be considered valid
# min_valid_kp = 15       # minimum number of valid keypoints to proceed with classification
# confidence_threshold = 0.7  # minimum confidence to consider prediction reliable

# @app.route('/predict_pose', methods=['POST'])
# def predict_pose():
#     try:
#         data = request.json
#         keypoints = data.get('keypoints')
#         frame_width = data.get('frameWidth')
#         frame_height = data.get('frameHeight')

#         if keypoints is None or len(keypoints) != 17:
#             return jsonify({'error': 'Expected 17 keypoints with [x, y, score].'}), 400
#         if frame_width is None or frame_height is None:
#             return jsonify({'error': 'Missing frameWidth or frameHeight.'}), 400

#         keypoints = np.array(keypoints, dtype=np.float32)

#         # Count valid keypoints by confidence score
#         valid_kp = np.sum(keypoints[:, 2] > kp_threshold)

#         if valid_kp < min_valid_kp:
#             return jsonify({
#                 'pose': None,
#                 'confidence': 0.0,
#                 'warning': f'Not enough valid keypoints detected ({valid_kp} < {min_valid_kp})',
#                 'keypoints': keypoints.tolist()
#             })

#         # Optionally normalize x and y by frame size if needed
#         # keypoints[:, 0] /= frame_width
#         # keypoints[:, 1] /= frame_height

#         # Reshape for model input: (1, 17, 3, 1)
#         input_data = keypoints.reshape(1, 17, 3, 1)

#         # Predict pose
#         prediction = model.predict(input_data)
#         pred_class = int(np.argmax(prediction))
#         confidence = float(np.max(prediction))
#         label = class_labels[pred_class]

#         warning = ""
#         if confidence < confidence_threshold:
#             warning = "Low confidence in pose detection"

#         return jsonify({
#             'pose': label,
#             'confidence': round(confidence, 3),
#             'warning': warning,
#             'keypoints': keypoints.tolist()
#         })

#     except Exception as e:
#         return jsonify({'error': str(e)}), 500





# @app.route('/')
# def health_check():
#     return "✅ Pose classification API is live!"

# if __name__ == '__main__':
#     import os
#     port = int(os.environ.get("PORT", 8080))
#     app.run(host='0.0.0.0', port=port)





# try 1

# from flask import Flask, request, jsonify
# import numpy as np
# import tensorflow as tf
# import pickle
# from flask_cors import CORS

# app = Flask(__name__)
# CORS(app)

# # Load the trained Keras model
# model = tf.keras.models.load_model("suryanamaskar4.h5")

# # Load the label encoder
# with open("label_encoder4.pkl", "rb") as f:
#     le = pickle.load(f)

# # Thresholds
# kp_threshold = 0.3       # Minimum score to consider a keypoint valid
# min_valid_kp = 15        # Required number of valid keypoints to proceed
# confidence_threshold = 0.7  # Minimum confidence to consider prediction reliable

# @app.route('/predict_pose', methods=['POST'])
# def predict_pose():
#     try:
#         data = request.json
#         keypoints = data.get('keypoints')
#         frame_width = data.get('frameWidth')
#         frame_height = data.get('frameHeight')

#         if keypoints is None or len(keypoints) != 17:
#             return jsonify({'error': 'Expected 17 keypoints with [x, y, score].'}), 400
#         if frame_width is None or frame_height is None:
#             return jsonify({'error': 'Missing frameWidth or frameHeight.'}), 400

#         keypoints = np.array(keypoints, dtype=np.float32)

#         # Count valid keypoints
#         valid_kp = np.sum(keypoints[:, 2] > kp_threshold)
#         if valid_kp < min_valid_kp:
#             return jsonify({
#                 'pose': None,
#                 'confidence': 0.0,
#                 'warning': f'Not enough valid keypoints detected ({valid_kp} < {min_valid_kp})',
#                 'keypoints': keypoints.tolist()
#             })

#         # Do NOT normalize — model is trained on raw keypoints
#         input_data = keypoints.reshape(1, 17, 3, 1)

#         # Predict pose
#         prediction = model.predict(input_data)
#         pred_class = int(np.argmax(prediction))
#         confidence = float(np.max(prediction))
#         label = le.inverse_transform([pred_class])[0]

#         warning = ""
#         if confidence < confidence_threshold:
#             warning = "Low confidence in pose detection"

#         return jsonify({
#             'pose': label,
#             'confidence': round(confidence, 3),
#             'warning': warning,
#             'keypoints': keypoints.tolist()
#         })

#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# @app.route('/')
# def health_check():
#     return "✅ Pose classification API is live!"

# if __name__ == '__main__':
#     import os
#     port = int(os.environ.get("PORT", 8080))
#     app.run(host='0.0.0.0', port=port)



# Test 1

from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import pickle
import time
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load model and label encoder
model = tf.keras.models.load_model("suryanamaskar6.h5")
with open("label_encoder6.pkl", "rb") as f:
    le = pickle.load(f)
class_labels = le.classes_

# Thresholds
kp_threshold = 0.3
min_valid_kp = 15
confidence_threshold = 0.7

# Pose timer tracking
pose_timers = {label: 0 for label in class_labels}
warning_flags = {label: False for label in class_labels}

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    return angle

@app.route('/predict_pose', methods=['POST'])
def predict_pose():
    try:
        data = request.json
        keypoints = data.get('keypoints')
        frame_width = data.get('frameWidth')
        frame_height = data.get('frameHeight')

        if keypoints is None or len(keypoints) != 17:
            return jsonify({'error': 'Expected 17 keypoints with [x, y, score].'}), 400
        if frame_width is None or frame_height is None:
            return jsonify({'error': 'Missing frameWidth or frameHeight.'}), 400

        keypoints = np.array(keypoints, dtype=np.float32)
        valid_kp = np.sum(keypoints[:, 2] > kp_threshold)
        if valid_kp < min_valid_kp:
            return jsonify({
                'pose': None,
                'confidence': 0.0,
                'warning': f'Not enough valid keypoints detected ({valid_kp} < {min_valid_kp})',
                'keypoints': keypoints.tolist()
            })

        input_data = keypoints.reshape(1, 17, 3, 1)
        prediction = model.predict(input_data)
        pred_class = int(np.argmax(prediction))
        confidence = float(np.max(prediction))
        label = le.inverse_transform([pred_class])[0]

        warning_text = ""
        now = time.time()

        # Reset timers for other poses
        for pose in pose_timers:
            if pose != label:
                pose_timers[pose] = 0
                warning_flags[pose] = False
        print(f"Hasta Uttanasana angle: {label}")
        if confidence >= confidence_threshold:
            # Hasta Uttanasana
            if label == "HastaUttasana":
                nose = keypoints[0][:2]
                hips = (keypoints[11][:2] + keypoints[12][:2]) / 2
                knees = (keypoints[13][:2] + keypoints[14][:2]) / 2
                angle = calculate_angle(nose, hips, knees)
                print(f"Hasta Uttanasana angle: {round(angle, 1)}°")
                if 170 < round(angle, 1) < 180:
                    warning_text = "⚠ Hasta Uttanasana: Lean back more!"
                    if pose_timers[label] == 0:
                        pose_timers[label] = now
                    elif now - pose_timers[label] >= 10:
                        warning_flags[label] = True
                else:
                    pose_timers[label] = 0
                    warning_flags[label] = False
                if warning_flags[label]:
                    warning_text = "⚠ Hasta Uttanasana: Lean back more!"

            # Padahastasana
            elif label == "Padahastasana":
                shoulders = (keypoints[5][:2] + keypoints[6][:2]) / 2
                hips = (keypoints[11][:2] + keypoints[12][:2]) / 2
                ankles = (keypoints[15][:2] + keypoints[16][:2]) / 2
                angle = calculate_angle(shoulders, hips, ankles)
                if round(angle, 1) < 100:
                    if pose_timers[label] == 0:
                        pose_timers[label] = now
                        warning_text = "⚠ Padahastasana: Bend down more!"
                    elif now - pose_timers[label] >= 10:
                        warning_flags[label] = True
                else:
                    pose_timers[label] = 0
                    warning_flags[label] = False
                if warning_flags[label]:
                    warning_text = "⚠ Padahastasana: Bend down more!"

            # Ashwa Sanchalanasana
            elif label == "ashwa _anchalanasana":
                shoulders = (keypoints[5][:2] + keypoints[6][:2]) / 2
                hips = (keypoints[11][:2] + keypoints[12][:2]) / 2
                nose = keypoints[0][:2]
                angle = calculate_angle(shoulders, hips, nose)
                if round(angle, 1) > 1.0:
                    if pose_timers[label] == 0:
                        pose_timers[label] = now
                    elif now - pose_timers[label] >= 10:
                        warning_flags[label] = True
                else:
                    pose_timers[label] = 0
                    warning_flags[label] = False
                if warning_flags[label]:
                    warning_text = "⚠ Ashwa Sanchalanasana: Lift your head!"

            # Kumbhakasana
            elif label == "Kumbhakasana":
                l_angle = calculate_angle(keypoints[11][:2], keypoints[13][:2], keypoints[15][:2])
                r_angle = calculate_angle(keypoints[12][:2], keypoints[14][:2], keypoints[16][:2])
                avg_angle = (l_angle + r_angle) / 2
                if round(avg_angle, 1) < 165:
                    if pose_timers[label] == 0:
                        pose_timers[label] = now
                    elif now - pose_timers[label] >= 10:
                        warning_flags[label] = True
                else:
                    pose_timers[label] = 0
                    warning_flags[label] = False
                if warning_flags[label]:
                    warning_text = "⚠ Kumbhakasana: Keep your legs straight!"

        elif confidence < confidence_threshold:
            warning_text = "⚠ Low confidence in pose detection"

        print(f"{warning_text}")

        return jsonify({
            'pose': label,
            'confidence': round(confidence, 3),
            'warning': warning_text,
            'keypoints': keypoints.tolist()
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def health_check():
    return "✅ Pose classification API is live!"

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)