import cv2
import time
import PoseModule

cap = cv2.VideoCapture('TestClip1.mp4')
previousTime = 0
detector = PoseModule.poseDetector()

while True:
  success, img = cap.read()
  img = detector.getPose(img)
  landmarkList = detector.getPosition(img)

  currentTime = time.time()
  fps = 1/(currentTime - previousTime)
  previousTime = currentTime

  cv2.putText(img, str(int(fps)), (70,50), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)
  cv2.imshow("Image", img)
    
  cv2.waitKey(1)