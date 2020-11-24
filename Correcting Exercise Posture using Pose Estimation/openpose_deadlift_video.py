import cv2
import math


protoFile = "/openpose_models/prototxt/pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
weightsFile = "/openpose_models/prototxt/pose/mpi/pose_iter_160000.caffemodel"


cap = cv2.VideoCapture(0)
hasFrame, frame = cap.read()

vid_writer = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame.shape[1],frame.shape[0]))

inWidth = 368
inHeight = 368
nPoints=15
threshold = 0.1

net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

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
    required_points=[0,1,2,3,4,8,9,10,14]
    for i in range(nPoints):
        probMap = output[0, i, :, :]
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
        x = (frameWidth * point[0]) / W
        y = (frameHeight * point[1]) / H
        if prob>threshold:
            if i in required_points:
                cv2.putText(frame, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)
            points.append((int(x), int(y)))
        else :
            points.append(None)

    POSE_PAIRS = [[0,1], [1,2], [2,3], [3,4],[1,14], [14,8], [8,9], [9,10] ]
    for pair in POSE_PAIRS:
        partA = pair[0]
        partB = pair[1]

        if points[partA] and points[partB]:
            cv2.line(frame, points[partA], points[partB], (0, 255, 255), 3, lineType=cv2.LINE_AA)
            cv2.circle(frame, points[partA], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
    def getAngle(a, b, c):
        try:
            ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
            return ang + 360 if ang < 0 else ang
        except:
            return 0

    def getDistance(a,b):
        math.sqrt(((b[0]-a[0])**2)+((b[1]-a[1])**2))

  
    angle_1=getAngle(points[0],points[1],points[2])
    angle_2=getAngle(points[2],points[3],points[4])
    angle_3=getAngle(points[2],points[14],points[8])
    angle_4=getAngle(points[8],points[9],points[10])

    angle_5=getAngle(points[1],points[2],points[8])
    angle_6=getAngle(points[8],points[9],points[10])
        

    if (int(angle_1) in range(148,178) and int(angle_2) in range(165,195) and int(angle_3) in range(154,184) and int(angle_4) in range(211,241)):
        print("deadlift start")
        cv2.putText(frame,"Deadlift Start Position",(0,100),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),2,cv2.LINE_AA)
    if (int(angle_5) in range(143,173) and int(angle_6) in range(165,195)):
        print("deadlift finish")
        cv2.putText(frame,"Deadlift Finish Position",(0,100),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),2,cv2.LINE_AA)
    cv2.imshow('Output-Skeleton', frame)
    
    vid_writer.write(frame)
vid_writer.release()