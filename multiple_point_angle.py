from __future__ import print_function
import sys
import cv2
import math
from random import randint
from scipy.spatial import distance as dist1
import time

TrackerType = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']


def createTrackerByName(trackerType):
    # Create a tracker based on tracker name
    if trackerType == TrackerType[0]:
        tracker = cv2.legacy.TrackerBoosting_create()
    elif trackerType == TrackerType[1]:
        tracker = cv2.legacy.TrackerMIL_create()
    elif trackerType == TrackerType[2]:
        tracker = cv2.legacy.TrackerKCF_create()
    elif trackerType == TrackerType[3]:
        tracker = cv2.legacy.TrackerTLD_create()
    elif trackerType == TrackerType[4]:
        tracker = cv2.legacy.TrackerMedianFlow_create()
    elif trackerType == TrackerType[5]:
        tracker = cv2.TrackerGOTURN_create()
    elif trackerType == TrackerType[6]:
        tracker = cv2.TrackerMOSSE_create()
    elif trackerType == TrackerType[7]:
        tracker = cv2.TrackerCSRT_create()
    else:
        tracker = None
        print('Incorrect tracker name')
        print('Available trackers are:')
        for t in TrackerType:
            print(t)

    return tracker


cap = cv2.VideoCapture("test.mp4")

success, frame = cap.read()

# SIZE
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

size = (frame_width, frame_height)

# VIDEO WRITER
result = cv2.VideoWriter('output_angle.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, size)

if not success:
    print('Failed to read video')
    sys.exit(1)

# Select boxes
bboxes = []
colors = []
mid_points = []

while True:
    bbox = cv2.selectROI('MultiTracker', frame)
    bboxes.append(bbox)
    colors.append((randint(0, 255), randint(0, 255), randint(0, 255)))
    print("Press q to quit selecting boxes and start tracking")
    print("Press any other key to select next object")
    k = cv2.waitKey(0) & 0xFF
    print(k)
    if k == 113:  # q is pressed
        break

trackerType = "CSRT"
createTrackerByName(trackerType)

# Create MultiTracker object
multiTracker = cv2.MultiTracker_create()
start_time = time.time()

# Initialize MultiTracker
for bbox in bboxes:
    multiTracker.add(createTrackerByName(trackerType), frame, bbox)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    success, boxes = multiTracker.update(frame)
    if success:
        for i, newbox in enumerate(boxes):
            p1 = (int(newbox[0]), int(newbox[1]))
            p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
            cv2.rectangle(frame, p1, p2, colors[i], 2, 1)
            mid_point = (int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2))
            cv2.circle(frame, mid_point, 5, (0, 0, 255), -1)
            cv2.putText(frame, "P" + str(i), (mid_point[0] + 5, mid_point[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                        (0, 0, 255), 2)
            mid_points.append(mid_point)

            if len(mid_points) == 3:
                cv2.line(frame, mid_points[0], mid_points[1], (0, 0, 255), 2)
                cv2.line(frame, mid_points[1], mid_points[2], (0, 255, 0), 2)
                x = mid_points[-2][0] - mid_points[-3][0]
                y = mid_points[-2][0] - mid_points[-1][0]
                z = mid_points[-2][1] - mid_points[-3][1]
                w = mid_points[-2][1] - mid_points[-1][1]

                p = x * y + z * w

                q = math.sqrt(pow(x, 2) + pow(z, 2))
                r = math.sqrt(pow(y, 2) + pow(w, 2))

                s = q * r

                angle = p / s

                theta = math.acos(angle)

                theta = theta * 180 / math.pi  # conversion from rad to degree

                theta = round(theta, 2)

                cv2.putText(frame, str(theta), (mid_points[-2][0] + 8, mid_points[-2][1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (255, 255, 255), 2, cv2.LINE_AA)

                mid_points = []
    else:
        # Tracking failure
        cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    result.write(frame)
    cv2.imshow('MultiTracker', frame)
    print("--- %s seconds ---" % (time.time() - start_time))
    if cv2.waitKey(1) & 0xFF == 27:  # Esc pressed
        break

cap.release()
result.release()

cv2.destroyAllWindows()