import socket
import cv2
import pickle
import struct
import numpy as np
import os
import time
import queue
import threading

PORT = 5050
FORMAT = 'ISO-8859-1'
DISCONNECT_MESSAGE = "!DISCONNECT"
SERVER = "192.168.0.160"
ADDR = (SERVER, PORT)

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(ADDR)

recvData = b''
payload_size = struct.calcsize("L")

count = 0
imagesCount = 0
offClientTimes = []


# bufferless VideoCapture
class VideoCapture:

  def __init__(self, name):
    self.cap = cv2.VideoCapture(name)
    self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    self.q = queue.Queue()
    self.isOpened = True
    t = threading.Thread(target=self._reader)
    t.daemon = True
    t.start()

  # read frames as soon as they are available, keeping only most recent one
  def _reader(self):
    while True:
      ret, frame = self.cap.read()
      if not ret:
        break
      if not self.q.empty():
        try:
          self.q.get_nowait()   # discard previous (unprocessed) frame
        except queue.Empty:
          pass
      self.q.put(frame)
      time.sleep(0.016)
    self.isOpened = False

  def read(self):
    return self.q.get()

  def getIsOpened(self):
      return self.isOpened

startTime = time.time()

for x in os.listdir("videos"):
    # x = "Kettlebell Training - 12697.mp4"
    image_path = "videos/" + x
    video = VideoCapture(image_path)
    while video.getIsOpened() and count < 100:
        frame = video.read()
        frame = cv2.resize(frame, (640, 480))
        if frame is not None:
            # Send encoded frame data from client to server
            data = pickle.dumps(frame, 5)
            message_size = struct.pack("L", len(data))
            client.sendall(message_size + data)
            offClientStart = time.time()

            # Wait for server to run inference and send processed frame back
            while len(recvData) < payload_size:
                recvData += client.recv(4096)
            packed_msg_size = recvData[:payload_size]
            recvData = recvData[payload_size:]
            msg_size = struct.unpack("L", packed_msg_size)[0]
            while len(recvData) < msg_size:
                recvData += client.recv(4096)
            offClientTimes.append((time.time() - offClientStart) * 1000)

            # Convert processed frame from byte data to frame
            frame_data = recvData[:msg_size]
            recvData = recvData[msg_size:]
            frame = pickle.loads(frame_data, fix_imports=True, encoding="bytes")

            if frame is not None:
                imagesCount += 1           

            # Display frame and updated previous frame
            cv2.imshow("Frame To Display", frame)
            cv2.waitKey(1)
        # time.sleep(0.08)
        count += 1
    count = 0

totalTime = time.time() - startTime
print("Total time elapsed (in ms): {}".format((totalTime)))
print("Images count is: {}".format(imagesCount))
print("Average inference time is: {}".format(sum(offClientTimes)/len(offClientTimes)))
print("Average FPS is: {}".format((imagesCount/totalTime))
