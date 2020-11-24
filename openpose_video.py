import cv2
import time


protoFile = "/openpose_models/prototxt/pose/body_25/pose_deploy.prototxt"
weightsFile = "/openpose_models/graph/pose/body_25/pose_iter_584000.caffemodel"


cap = cv2.VideoCapture(0)
hasFrame, frame = cap.read()
fps = cap.get( cv2.CAP_PROP_FPS )
vid_writer = cv2.VideoWriter('sample_video.mp4',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame.shape[1],frame.shape[0]))


inWidth = 368
inHeight = 368
nPoints=19
threshold = 0.1
fps_time = 0

net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

while cv2.waitKey(1) < 0:
    hasFrame, frame = cap.read()
    if not hasFrame:
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
            cv2.circle(frame, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(frame, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)
            points.append((int(x), int(y)))
        else :
            points.append(None)

    POSE_PAIRS = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,8],[8,9],[8,12],[9,10],[10,11],[12,13],[13,14],[15,17],[15,0],[0,16],[16,18] ]
    for pair in POSE_PAIRS:
        partA = pair[0]
        partB = pair[1]

        if points[partA] and points[partB]:
            cv2.line(frame, points[partA], points[partB], (0, 255, 255), 3, lineType=cv2.LINE_AA)
            cv2.circle(frame, points[partA], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.circle(frame, points[partB], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
    cv2.putText(frame,
                    "FPS: %f" % (1.0 / (time.time() - fps_time)),
                    (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)
    cv2.imshow('Output-Skeleton', frame)
    fps_time = time.time()
    if cv2.waitKey(int(1000/fps)) == 27:
        break  


    
    vid_writer.write(frame)
cv2.destroyAllWindows()
vid_writer.release()











