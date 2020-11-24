import cv2


protoFile = "/openpose_models/prototxt/pose/body_25/pose_deploy.prototxt"
weightsFile = "/openpose_models/graph/pose/body_25/pose_iter_584000.caffemodel"

net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

frame = cv2.imread("img.jpg")

frameWidth = frame.shape[1]
frameHeight = frame.shape[0]

inWidth = 368
inHeight = 368

inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)

net.setInput(inpBlob)

output = net.forward()

H = output.shape[2]
W = output.shape[3]

nPoints=19
threshold = 0.1

points = []

for i in range(nPoints):
    probMap = output[0, i, :, :]
    minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
    x = (frameWidth * point[0]) / W
    y = (frameHeight * point[1]) / H
    if prob>threshold:
        cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
        cv2.putText(frame, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, lineType=cv2.LINE_AA)
        points.append((int(x), int(y)))
    else :
        points.append(None)
        
cv2.imwrite('Output-Keypoints-full_pose.jpg', frame)

POSE_PAIRS = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,8],[8,9],[8,12],[9,10],[10,11],[12,13],[13,14],[15,17],[15,0],[0,16],[16,18] ]

for pair in POSE_PAIRS:
    partA = pair[0]
    partB = pair[1]

    if points[partA] and points[partB]:
        cv2.line(frame, points[partA], points[partB], (0, 255, 255), 2)
        # cv2.circle(frame, points[partA], 5, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
cv2.imwrite('Output-Skeleton-full_pose.jpg', frame)
