import tensorflow as tf
import cv2
import os
import time

def movenet(input_image):
        """Runs detection on an input image.

        Args:
        input_image: A [1, height, width, 3] tensor represents the input image
            pixels. Note that the height/width should already be resized and match the
            expected input resolution of the model before passing into this function.

        Returns:
        A [1, 1, 17, 3] float numpy array representing the predicted keypoint
        coordinates and scores.
        """
        # TF Lite format expects tensor type of uint8.
        input_image = tf.cast(input_image, dtype=tf.uint8)
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.set_tensor(input_details[0]['index'], input_image.numpy())
        # Invoke inference.
        interpreter.invoke()
        # Get the model prediction.
        keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
        return keypoints_with_scores

def main():
    # Initialize the TFLite interpreter
    interpreter = tf.lite.Interpreter(model_path="lite-model_movenet_singlepose_lightning_tflite_int8_4.tflite")
    input_size = 192
    interpreter.allocate_tensors()

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    # read in the image
    success, img = cap.read()
    # update frame
    cv2.putText(img, f'FPS: {int(fps)}', (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 3)
    cv2.imshow("Frame", img)
    cv2.waitKey(1)

if __name__ == "__main__":
    main()