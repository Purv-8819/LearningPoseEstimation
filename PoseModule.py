import cv2
import mediapipe as mp
import time


class poseDetector():
  def __init__(self):
    
    self.mpDraw = mp.solutions.drawing_utils
    self.mpPose = mp.solutions.pose
    self.pose = self.mpPose.Pose()

  def getPose(self, img, draw = True):
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    self.results = self.pose.process(imgRGB)
    if self.results.pose_landmarks:
      if draw:
        self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
    return img


  def getPosition(self, img, draw = True):
    landmarkList = []
    if self.results.pose_landmarks:
      for id, landmark in enumerate(self.results.pose_landmarks.landmark):
        h, w, c = img.shape
        screenx, screeny = int(landmark.x*w), int(landmark.y*h)
        landmarkList.append([id, screenx, screeny])
        if draw:
          cv2.circle(img, (screenx, screeny), 10, (0,255,255), cv2.FILLED)                        
    return landmarkList


def main():
  cap = cv2.VideoCapture('TestClip1.mp4')
  previousTime = 0
  detector = poseDetector()

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


if __name__ == "__main__":
  main()