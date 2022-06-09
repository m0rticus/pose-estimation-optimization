import socket
import threading
import cv2
import struct
import pickle
import time

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

# Dictionary for fast drawing access
EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}

module = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
PORT = 5050
SERVER = socket.gethostbyname(socket.gethostname())
ADDR = (SERVER, PORT)
FORMAT = 'ISO-8859-1'
DISCONNECT_MESSAGE = "!DISCONNECT"

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(ADDR)

# Load movenet model from module and run inference
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
    model = module.signatures['serving_default']

    # SavedModel format expects tensor type of int32.
    input_image = tf.cast(input_image, dtype=tf.int32)
    # Run model inference.
    outputs = model(input_image)
    # Output is a [1, 1, 17, 3] tensor.
    keypoints_with_scores = outputs['output_0'].numpy()
    return keypoints_with_scores

# Found on StackOverflow
# Function to calculate Structural Similarity Index for comparing images
def ssim(i1, i2):
    c1 = 6.5025
    c2 = 58.5225
    # INITS
    I1 = np.float32(i1) # cannot calculate on one byte large values
    I2 = np.float32(i2)
    I2_2 = I2 * I2 # I2^2
    I1_2 = I1 * I1 # I1^2
    I1_I2 = I1 * I2 # I1 * I2
    # END INITS
    # PRELIMINARY COMPUTING
    mu1 = cv2.GaussianBlur(I1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(I2, (11, 11), 1.5)
    mu1_2 = mu1 * mu1
    mu2_2 = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_2 = cv2.GaussianBlur(I1_2, (11, 11), 1.5)
    sigma1_2 -= mu1_2
    sigma2_2 = cv2.GaussianBlur(I2_2, (11, 11), 1.5)
    sigma2_2 -= mu2_2
    sigma12 = cv2.GaussianBlur(I1_I2, (11, 11), 1.5)
    sigma12 -= mu1_mu2
    t1 = 2 * mu1_mu2 + c1
    t2 = 2 * sigma12 + c2
    t3 = t1 * t2                    # t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))
    t1 = mu1_2 + mu2_2 + c1
    t2 = sigma1_2 + sigma2_2 + c2
    t1 = t1 * t2                    # t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))
    ssim_map = cv2.divide(t3, t1)    # ssim_map =  t3./t1;
    mssim = cv2.mean(ssim_map)       # mssim 
    return mssim

input_size = 256
keypoints_with_scores = []

# Threaded function to handle unique clients
# There's really only one client, but it could work with more
def handle_client(conn, addr):
    print(f"Handling client {addr}")
    inferences = []
    data = b''
    count = 0
    previousFrame = None
    payload_size = struct.calcsize("L")
    # While client is connected
    try:
        while True:
            # Receive data from connected client
            while len(data) < payload_size:
                data += conn.recv(4096)
            packed_msg_size = data[:payload_size]
            data = data[payload_size:]
            msg_size = struct.unpack("L", packed_msg_size)[0]
            while len(data) < msg_size:
                data += conn.recv(4096)
                
            # Convert data into frame object for inference
            frame_data = data[:msg_size]
            data = data[msg_size:]

            # print(frame_data)
            frame = pickle.loads(frame_data, fix_imports=True, encoding="bytes")

            if frame is not None:
                # print("Loaded frame")
                original_frame = frame
                startInference = time.time()
                if count == 0:
                    previousFrame = frame

                # Calculate difference between frames and whether it's unique enough to process
                frame_diff = ssim(previousFrame, frame)
                # print("\frame_diff: R {}% G {}% B {}%".format(round(frame_diff[2] * 100, 2), 
                        # round(frame_diff[1] * 100, 2),
                        # round(frame_diff[0] * 100, 2)))
                if (frame_diff[2] * 100 * .33 + frame_diff[1] * 100 * .33 + frame_diff[0] * 100 * .33) < 98 or count == 0:

                    # Resize image to input 256x256 for model inference
                    input_image = tf.expand_dims(frame, axis=0)
                    input_image = tf.image.resize_with_pad(input_image, input_size, input_size)
                    keypoints_with_scores = movenet(input_image)

                    inferences.append((time.time() - startInference) * 1000)
                    
                    previousFrame = original_frame

                # Visualization functions
                draw_edges(original_frame, keypoints_with_scores, EDGES, 0.3)
                draw_keypoints(original_frame, keypoints_with_scores, 0.3)
                                    
                # Send processed image back to client for display
                framedData = pickle.dumps(original_frame)
                message_size = struct.pack("L", len(framedData))
                conn.sendall(message_size + framedData)
                count += 1

            else:
                print("Failed to load frame")
                break

            # msg = conn.recv(1024).decode(FORMAT)
            # if msg:
                # if msg == DISCONNECT_MESSAGE:
                    # break
                # print(msg)
                # conn.send("Message received".encode(FORMAT))
                
    except ConnectionError:
        print("Average inference time is: {}".format(sum(inferences)/len(inferences)))
        print("Disconnecting...")
        conn.close()

# Helper function to visualize keypoints
# Taken from StackOverflow / YouTube tutorial
def draw_keypoints(frame, keypoints, confidence):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))

    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence:
            cv2.circle(frame, (int(kx), int(ky)), 4, (0,255,0), -1)

# Helper function to visualize edges
# Taken from StackOverflow / YouTube tutorial
def draw_edges(frame, keypoints, edges, confidence):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))

    for edge, color in edges.items():
        p1, p2 = edge
        y1,x1,c1 = shaped[p1]
        y2,x2,c2 = shaped[p2]

        if c1 > confidence and c2 > confidence:
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)

def start():
    server.listen()
    print("Listening!")
    while True:
        conn, addr = server.accept()
        thread = threading.Thread(target=handle_client, args=(conn, addr))
        print("Thread started!")
        thread.start()
    
print("Server starting...")
start()