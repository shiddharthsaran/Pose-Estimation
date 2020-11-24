
from imutils.video import VideoStream
import imagezmq
import argparse
import socket
import time
import cv2
import json

fps_time = 0
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--server-port", required=True,
	help="Port is address of the service within the System.")
args = vars(ap.parse_args())

# sender = imagezmq.ImageSender(connect_to="tcp://2.tcp.ngrok.io:{}".format(
# 	args["server_port"]))
sender = imagezmq.ImageSender(connect_to="tcp://127.0.0.1:5555")
rpiName = socket.gethostname()

cap = cv2.VideoCapture(0)
fps = cap.get( cv2.CAP_PROP_FPS )
def vid_wri(frame):
	global vid_writer 

	fourcc = cv2.VideoWriter_fourcc(*'MPEG')
	vid_writer= cv2.VideoWriter('sample_video-output.mp4',fourcc, fps,  (frame.shape[1],frame.shape[0]))
	
bool_val=True

while True:
	ret, frame = cap.read()
	try:
		result=sender.send_image(rpiName, frame)
	except:
		break
	strs = result.decode("ascii")
	points=json.loads(strs)
	indices = [i for i, item in enumerate(points) if item == 13]
	
	if len(indices)>0:
		for ind in indices:
			points[ind]=None

	for i,point in enumerate(points):
		try:
			cv2.putText(frame, "{}".format(i), tuple(point), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)
		except:
			pass
	POSE_PAIRS = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,8],[8,9],[8,12],[9,10],[10,11],[12,13],[13,14] ]
	for pair in POSE_PAIRS:
		partA = pair[0]
		partB = pair[1]
		if points[partA] and points[partB]:
			cv2.line(frame, tuple(points[partA]), tuple(points[partB]), (0, 255, 255), 3, lineType=cv2.LINE_AA)
			cv2.circle(frame, tuple(points[partA]), 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
			cv2.circle(frame, tuple(points[partB]), 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
	cv2.putText(frame,
                    "Client-FPS: %f" % (1.0 / (time.time() - fps_time)),
                    (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)
	cv2.imshow('frame',frame)

	if bool_val:
		vid_wri(frame)
	else:
		pass
	bool_val=False
	vid_writer.write(frame)
	fps_time = time.time()
	if cv2.waitKey(int(1000/fps)) & 0xFF == ord("q"):
	    break
try:
	cap.release()
	cv2.destroyAllWindows()
	vid_writer.release()
except:
	pass
