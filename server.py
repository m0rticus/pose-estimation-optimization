import socket
import threading
import cv2
import struct
import pickle

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
FORMAT = 'utf-8'
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

input_size = 256

# Threaded function to handle unique clients
# There's really only one client, but it could work with more
def handle_client(conn, addr):
    print(f"Handling client {addr}")
    data = b''
    payload_size = struct.calcsize("L")
    # While client is connected
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
        frame = pickle.loads(frame_data)

        print("Loaded frame")
        if frame is None:
            break

        original_frame = frame

        # Resize image to input 256x256 for model inference
        input_image = tf.expand_dims(frame, axis=0)
        input_image = tf.image.resize_with_pad(input_image, input_size, input_size)
        keypoints_with_scores = movenet(input_image)

        # Visualization functions
        draw_edges(original_frame, keypoints_with_scores, EDGES, 0.5)
        draw_keypoints(original_frame, keypoints_with_scores, 0.5)

        # Send processed image back to client for display
        framedData = pickle.dumps(original_frame)
        message_size = struct.pack("L", len(framedData))
        conn.sendall(message_size + framedData)

        # cv2.imshow("Pose Estimation", original_frame)
        # cv2.waitKey(1)

        # msg = conn.recv(1024).decode(FORMAT)
        # if msg:
            # if msg == DISCONNECT_MESSAGE:
                # break
            # print(msg)
            # conn.send("Message received".encode(FORMAT))

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