
import numpy as np
import imagezmq
import cv2

import json



protoFile = "/openpose_models/prototxt/pose/body_25/pose_deploy.prototxt"
weightsFile = "/openpose_models/graph/pose/body_25/pose_iter_584000.caffemodel"


inWidth = 368
inHeight = 368
nPoints=15
threshold = 0.1

imageHub = imagezmq.ImageHub()

net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)






while True:
	(rpiName, frame) = imageHub.recv_image()

	if not rpiName:
		cv2.waitKey()
		break
	
	frameWidth = frame.shape[1]
	frameHeight = frame.shape[0]
	
	inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)

	net.setInput(inpBlob)
	output = net.forward()
	H = output.shape[2]
	W = output.shape[3]

	points = []
	for i in range(nPoints):
		probMap = output[0, i, :, :]
		minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
		x = (frameWidth * point[0]) / W
		y = (frameHeight * point[1]) / H
		if prob>threshold:

			points.append((int(x), int(y)))
		else :
			points.append(13)

	my_json_string = json.dumps(points)
	imageHub.send_reply(my_json_string.encode('ascii'))

