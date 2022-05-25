# MODULE NAME : MPKIT_CC
# AUTHOR : CHANCHAL ROY
# VERSION : 0.0.4
try:
    from mediapipe.python.solutions import (
    hands,
    face_mesh,
    face_detection,
    pose,
    drawing_utils,
    drawing_styles)
    from mediapipe.python.solutions.drawing_utils import DrawingSpec
    from cv2 import (
    CAP_DSHOW,
    CAP_PROP_FOURCC,
    CAP_PROP_FPS,
    CAP_PROP_FRAME_HEIGHT,
    CAP_PROP_FRAME_WIDTH,
    COLOR_BGR2RGB,
    VideoCapture,
    VideoWriter_fourcc,
    cvtColor,
    putText,
    rectangle,
    FONT_HERSHEY_COMPLEX)
    
except ImportError as i: print(i)

# BASIC INFORMATION CONSTANTS
__author__ = "Chanchal Roy"
__version__ = "0.0.4"
__func__ = ["init","show_FPS","find_Hands","find_face","face_mesh"]
__module__ = f"mpkit_cc | {__version__}"
__sub_module__ = ["opencv-python","mediapipe"]

# DEFAULT COLOUR VALUES (CONSTANTS)
RED = (0,0,255)
YELLOW = (0,255,255)
GREEN = (0,255,0)
CYAN = (255,255,0)
BLUE = (255,0,0)
PINK = (255,0,255)
WHITE = (255,255,255)
BLACK = (0,0,0)
GRAY = (80,80,80)

class Mptools:
  def __init__(self,cam_index:int = 0,win_width:int = 640,win_height:int = 360,cam_fps:int = 30,hand_no:int = 1,face_no:int = 1):
      self.mp_hand = hands.Hands(False,hand_no,1,.5,.5)
      self.mp_face = face_detection.FaceDetection(.5,0)
      self.mp_mesh = face_mesh.FaceMesh(False,face_no,True,.5,.5)
      self.mp_pose = pose.Pose(False,1,True,False,True,.5,.5)
      self.i = cam_index
      self.w = win_width
      self.h = win_height
      self.f = cam_fps

  def init(self):
    """Initiate the camera.
    
    Parameter
    =========
    It has no parameter.
    
    Return
    ======
    It returns the image grabed by the camera else for any exception returns none."""
    try:
      self.cam = VideoCapture(self.i,CAP_DSHOW)
      self.cam.set(CAP_PROP_FRAME_WIDTH,self.w)
      self.cam.set(CAP_PROP_FRAME_HEIGHT,self.h)
      self.cam.set(CAP_PROP_FPS,self.f)
      self.cam.set(CAP_PROP_FOURCC,VideoWriter_fourcc(*'MJPG'))
      return self.cam

    except Exception as e:
      print(e)
      return None

  def show_FPS(self,image_bgr,fps_rate,fore_bg:tuple = YELLOW,back_bg:tuple = RED):
    """Shows the FPS of the window.
    
    Parameter
    =========
    ``image_bgr`` : BGR image grabed by the camera.\n
    ``fps_rate`` : FPS rate of the window.\n
    ``fore_bg`` : Colour of the text. Default to Yellow.\n
    ``back_bg`` : Colour of the Background. Default to Red.
    
    Return
    ======
    It returns ``True`` else for any exception returns ``False``."""
    try:
      rectangle(image_bgr,(15,20),(155 ,60),back_bg,-1)
      putText(image_bgr,f"FPS: {fps_rate}",(20,50),FONT_HERSHEY_COMPLEX,1,fore_bg,2)
      return True

    except Exception as e:
      print(e)
      return False

  def find_Hands(self,image_bgr,hand_connection:bool = False,show_detect:bool = True):
    """It finds your hands and shows hand landmarks.
    
    Parameter :
    ==========
    ``image_bgr`` = BGR Image which you want to find the hands it.\n
    ``hand_connection`` = Set ``True`` if want to connect all hand landmarks else set ``False``. Default is ``False``.\n
    ``show_detect`` = Set ``True`` if visually show the detection else set it ``False``. Default set it ``True``.
    
    Return :
    ========
    It retuns all hand landmarks of both hands with index else for any exception it return empty string."""
    try:
      my_data = []
      right_hand = []
      left_hand = []
      frame_RGB = cvtColor(image_bgr,COLOR_BGR2RGB)
      results = self.mp_hand.process(frame_RGB)

      if results.multi_handedness != None:
        for id, hand_handedness in enumerate(results.multi_handedness):
          for id,hand_name in enumerate(hand_handedness.classification):

            if hand_name.label != None:
              if hand_name.label == "Left":
                if results.multi_hand_landmarks !=  None:
                  for id,hand_landmark in enumerate(results.multi_hand_landmarks):
                    if show_detect:
                      if hand_connection: drawing_utils.draw_landmarks(image_bgr,hand_landmark,hands.HAND_CONNECTIONS,DrawingSpec(RED),DrawingSpec(GREEN))
                      else: drawing_utils.draw_landmarks(image_bgr,hand_landmark,landmark_drawing_spec=DrawingSpec(RED))

                    for id,landMark in enumerate(hand_landmark.landmark):
                      right_hand.append((id,int(landMark.x*640),int(landMark.y*360),round(landMark.z,2)))

              elif hand_name.label == "Right":
                if results.multi_hand_landmarks !=  None:
                  for id,hand_landmark in enumerate(results.multi_hand_landmarks):
                    if show_detect:
                      if hand_connection: drawing_utils.draw_landmarks(image_bgr,hand_landmark,hands.HAND_CONNECTIONS,DrawingSpec(RED),DrawingSpec(GREEN))
                      else: drawing_utils.draw_landmarks(image_bgr,hand_landmark,landmark_drawing_spec=DrawingSpec(RED))

                    for id,landMark in enumerate(hand_landmark.landmark):
                      left_hand.append((id,int(landMark.x*640),int(landMark.y*360),round(landMark.z,2)))
              else: pass

        my_data = right_hand,left_hand

      return my_data

    except Exception as e:
      print(e)
      return None

  def find_face(self,image_bgr,show_detect:bool = True):
    """It finds your face and shows face landmarks.
    
    Parameter :
    ==========
    ``image_bgr`` : BGR Image which you want to find the hands it.\n
    ``show_detect`` = Set ``True`` if visually show the detection else set it ``False``. Default set it ``True``.
    
    Return :
    ========
    It retuns all hand landmarks of both hands with index else for any exception it return empty string."""
    try:
      face_landmark = []
      frame_RGB = cvtColor(image_bgr,COLOR_BGR2RGB)
      results = self.mp_face.process(frame_RGB)

      if results.detections != None:
        for id,detection in enumerate(results.detections):

          if show_detect:
            drawing_utils.draw_detection(image_bgr,detection,DrawingSpec(RED),DrawingSpec(GREEN))

          boundry_box_landmark = detection.location_data.relative_bounding_box
          boundry_box = (
            int(boundry_box_landmark.xmin * self.w),
            int(boundry_box_landmark.ymin * self.h),
            int(boundry_box_landmark.width * self.w),
            int(boundry_box_landmark.height * self.h))

          for id,face_landmarks in enumerate(detection.location_data.relative_keypoints):
            face_landmark.append((id,int(face_landmarks.x * self.w), int(face_landmarks.y * self.h)))

      face_data = boundry_box,face_landmark

      return face_data

    except Exception as e:
      print(e)
      return None

  def face_mesh(self,image_bgr,face_connection:bool = False,show_detect:bool = True):
    """It finds your face and shows full face landmarks.
    
    Parameter :
    ==========
    ``image_bgr`` = BGR Image which you want to find the hands it.\n
    ``face_connection`` = Set ``True`` if want to connect all face landmarks else set ``False``. Default is ``False``.\n
    ``show_detect`` = Set ``True`` if visually show the detection else set it ``False``. Default set it ``True``.
    
    Return :
    ========
    It retuns all face landmarks  with index else for any exception it return empty string."""
    try:
      my_face = []
      image_RGB = cvtColor(image_bgr, COLOR_BGR2RGB)
      results = self.mp_mesh.process(image_RGB)

      if results.multi_face_landmarks != None:
        for id,face_landmarks in enumerate(results.multi_face_landmarks):
          for id,face_landmark in enumerate(face_landmarks.landmark):

            my_face.append((id,int(face_landmark.x * self.w), int(face_landmark.y * self.h), round(face_landmark.z,2)))

          if show_detect:
            if face_connection:
              drawing_utils.draw_landmarks(image_bgr,face_landmarks,face_mesh.FACEMESH_TESSELATION,None,drawing_styles.get_default_face_mesh_tesselation_style())

            else:
              drawing_utils.draw_landmarks(image_bgr,face_landmarks,face_mesh.FACEMESH_FACE_OVAL,None,DrawingSpec(GREEN,1))

              drawing_utils.draw_landmarks(image_bgr,face_landmarks,face_mesh.FACEMESH_LEFT_EYE,None,DrawingSpec(GREEN,1))

              drawing_utils.draw_landmarks(image_bgr,face_landmarks,face_mesh.FACEMESH_RIGHT_EYE,None,DrawingSpec(GREEN,1))

              drawing_utils.draw_landmarks(image_bgr,face_landmarks,face_mesh.FACEMESH_RIGHT_EYEBROW,None,DrawingSpec(GREEN,1))

              drawing_utils.draw_landmarks(image_bgr,face_landmarks,face_mesh.FACEMESH_LEFT_EYEBROW,None,DrawingSpec(GREEN,1))

              drawing_utils.draw_landmarks(image_bgr,face_landmarks,face_mesh.FACEMESH_LIPS,None,DrawingSpec(GREEN,1))

              drawing_utils.draw_landmarks(image_bgr,face_landmarks,face_mesh.FACEMESH_LEFT_IRIS,None,DrawingSpec(RED))

              drawing_utils.draw_landmarks(image_bgr,face_landmarks,face_mesh.FACEMESH_RIGHT_IRIS,None,DrawingSpec(RED))

      return my_face

    except Exception as e:
      print(e)
      return None

  def find_pose(self,image_bgr,body_connection:bool = False,show_detect:bool = True):
    """It finds your full body pose and shows body landmarks.
    
    Parameter :
    ==========
    ``image_bgr`` = BGR Image which you want to find the hands it.\n
    ``body_connection`` = Set ``True`` if want to connect all body landmarks else set ``False``. Default is ``False``.\n
    ``show_detect`` = Set ``True`` if visually show the detection else set it ``False``. Default set it ``True``.
    
    Return :
    ========
    It retuns all body landmarks  with index else for any exception it return empty string."""
    try:
      my_data = []
      image_RGB = cvtColor(image_bgr,COLOR_BGR2RGB)
      results = self.mp_pose.process(image_RGB)

      if results.pose_landmarks != None:
        for id,pose in enumerate(results.pose_landmarks.landmark):
          my_data.append((id,int(pose.x * self.w),int(pose.y * self.h)))
          if show_detect:
            if body_connection:
              drawing_utils.draw_landmarks(image_bgr,results.pose_landmarks,pose.POSE_CONNECTIONS,DrawingSpec(RED),DrawingSpec(GREEN))
            
            else:
              drawing_utils.draw_landmarks(image_bgr,results.pose_landmarks,None,DrawingSpec(RED),None)

      return my_data

    except Exception as e:
      print(e)
      return None

if __name__ == '__main__':
  pass