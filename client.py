import socket
import cv2
import pickle
import struct

PORT = 5050
FORMAT = 'utf-8'
DISCONNECT_MESSAGE = "!DISCONNECT"
SERVER = "192.168.0.160"
ADDR = (SERVER, PORT)

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(ADDR)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, 192)
cap.set(4, 192)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

while True:
    if cap.isOpened():
        ret, frame = cap.read()
        # frame = cv2.resize(frame, (224, 224))
        # cv2.imshow('Hello', frame)
        data = pickle.dumps(frame)
        message_size = struct.pack("L", len(data))
        client.sendall(message_size + data)
    cv2.waitKey(1)

# def send(msg):
#     message = msg.encode(FORMAT)
#     client.send(message)
#     print(client.recv(1024).decode(FORMAT))

# send("Hello 1")
# send("Hello 2")
# send("Hello 3")
# send(DISCONNECT_MESSAGE)