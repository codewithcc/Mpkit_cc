# MODULE NAME : MPKIT_CC
# AUTHOR : CHANCHAL ROY
# VERSION : 0.0.5

try:
  from mediapipe.python.solutions import (
    hands,
    face_mesh,
    face_detection,
    pose)

  from mediapipe.python.solutions.drawing_utils import (
    DrawingSpec,
    _normalized_to_pixel_coordinates,
    draw_landmarks,
    draw_detection)

  from mediapipe.python.solutions.drawing_styles import (
    get_default_hand_landmarks_style,
    get_default_hand_connections_style,
    get_default_face_mesh_contours_style,
    get_default_face_mesh_iris_connections_style,
    get_default_face_mesh_tesselation_style,
    get_default_pose_landmarks_style
  )

  from cv2 import (
    CAP_DSHOW,
    CAP_PROP_FOURCC,
    CAP_PROP_FPS,
    CAP_PROP_FRAME_HEIGHT,
    CAP_PROP_FRAME_WIDTH,
    COLOR_BGR2RGB,
    COLOR_GRAY2RGB,
    COLOR_GRAY2BGR,
    COLOR_RGB2BGR,
    VideoCapture,
    circle,
    VideoWriter_fourcc,
    cvtColor,
    putText,
    rectangle,
    line,
    FONT_HERSHEY_COMPLEX)
    
except ImportError as i: print(i)

# BASIC INFORMATION CONSTANTS
__author__ = "Chanchal Roy"
__version__ = "0.0.5"
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

THUMB = [(0,1),(1,2),(2,3),(3,4)]
INDEX = [(5,6),(6,7),(7,8)]
MIDDLE = [(9,10),(10,11),(11,12)]
RING = [(13,14),(14,15),(15,16)]
PINKEY = [(17,18),(18,19),(19,20)]
PALM = [(0,5),(5,9),(9,13),(13,17),(0,17)]

FINGERS = [THUMB,INDEX,MIDDLE,RING,PINKEY,PALM]

class Mptools:
  
  def __init__(
    self,
    image_mode:bool = False,
    cam_index:int = 0,
    win_width:int = 640,
    win_height:int = 360,
    cam_fps:int = 30,
    hand_no:int = 2,
    face_no:int = 1,
    tol1:float = 0.5,
    tol2:float = 0.5):

      self.mp_hand = hands.Hands(
        static_image_mode = image_mode,
        max_num_hands = hand_no,
        model_complexity = 1,
        min_detection_confidence = tol1,
        min_tracking_confidence = tol2
      )
      self.mp_face = face_detection.FaceDetection(
        min_detection_confidence = tol1,
        model_selection = 0
      )
      self.mp_mesh = face_mesh.FaceMesh(
        static_image_mode = image_mode,
        max_num_faces = face_no,
        refine_landmarks = True,
        min_detection_confidence = tol1,
        min_tracking_confidence = tol2
      )
      self.mp_pose = pose.Pose(
        static_image_mode = image_mode,
        model_complexity = 1,
        smooth_landmarks = True,
        enable_segmentation = True,
        smooth_segmentation = True,
        min_detection_confidence = tol1,
        min_tracking_confidence = tol2
      )
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
      camera = VideoCapture(self.i,CAP_DSHOW)
      camera.set(CAP_PROP_FRAME_WIDTH,self.w)
      camera.set(CAP_PROP_FRAME_HEIGHT,self.h)
      camera.set(CAP_PROP_FPS,self.f)
      camera.set(CAP_PROP_FOURCC,VideoWriter_fourcc(*'MJPG'))
      return camera

    except Exception as e:
      print(e)
      return None

  def show_FPS(self,image,mode:str = "BGR",fps_rate:int = 0,fore_bg:tuple = YELLOW,back_bg:tuple = RED):
    """Shows the FPS of the window.
    
    Parameter
    =========
    ``image`` = Image which you want to find the hands it.\n
    ``mode`` = In which image type format you have.
    Set ``BGR`` for BGR type image, set ``RGB`` for RGB type image and ``B&W`` for gray Scale image.
    Default is ``BGR``.\n
    ``fps_rate`` : FPS rate of the window.\n
    ``fore_bg`` : Colour of the text. Default to Yellow.\n
    ``back_bg`` : Colour of the Background. Default to Red.
    
    Return
    ======
    It returns ``Image`` else ``None``."""
    try:
      if mode == "BGR":
        image_BGR = image
      if mode == "RGB":
        image_BGR = cvtColor(image,COLOR_RGB2BGR)
      if mode == "B&W":
        image_BGR = cvtColor(image,COLOR_GRAY2BGR)

      rectangle(image_BGR,(15,20),(155 ,60),back_bg,-1)
      putText(image_BGR,f"FPS: {fps_rate}",(20,50),FONT_HERSHEY_COMPLEX,1,fore_bg,2)
      return image_BGR

    except Exception as e:
      print(e)
      return None

  def find_Hands(self,image,mode:str = "BGR",hand_connection:bool = False,show_detect:bool = True,detection_style:int = 0):
    """It finds your hands and shows hand landmarks.
    
    Parameter :
    ==========
    ``image`` = Image which you want to find the hands it.\n
    ``mode`` = In which image type format you have.
    Set ``BGR`` for BGR type image, set ``RGB`` for RGB type image and ``B&W`` for gray Scale image.
    Default is ``BGR``.\n
    ``hand_connection`` = Set ``True`` if want to connect all hand landmarks else set ``False``. Default is ``False``.\n
    ``show_detect`` = Set ``True`` if visually show the detection else set it ``False``. Default set it ``True``.\n
    ``detection_style`` = Set ``0`` for the custom drawing style or set ``1`` for the default mediapipe detection style.
    
    Return :
    ========
    It returns a tuple containing hand landmarks, hand type and detection percentage of both hand."""
    try:
      my_hands = []
      hands_type = []
      hands_score = []

      def default_hand_draw_style(image,hand_data:list,hand_type:list):
        h,w,_ = image.shape
        if hand_data != None:
          for hand,handtp in zip(hand_data,hand_type):
            if handtp == "Right":
              lm_clr = GREEN
            if handtp == "Left":
              lm_clr = RED
            for idx in range(0,len(hand)):
              coordinate = _normalized_to_pixel_coordinates(hand[idx][0],hand[idx][1],w,h)
              if coordinate != None:
                circle(image,coordinate,4,WHITE,-1)
                circle(image,coordinate,3,lm_clr,-1)
        return image

      def default_hand_connection_style(image,hand_data:list):
        h,w,_ = image.shape
        if hand_data != None:
          for hand in hand_data:
            for finger in FINGERS:
              for each_finger in finger:
                pt1 = _normalized_to_pixel_coordinates(hand[each_finger[0]][0],hand[each_finger[0]][1],w,h)
                pt2 = _normalized_to_pixel_coordinates(hand[each_finger[1]][0],hand[each_finger[1]][1],w,h)
                line(image,pt1,pt2,WHITE,2)
        return image

      if mode == "BGR":
        image_RGB = cvtColor(image,COLOR_BGR2RGB)
      if mode == "RGB":
        image_RGB = image
      if mode == "B&W":
        image_RGB = cvtColor(image,COLOR_GRAY2RGB)
      results = self.mp_hand.process(image_RGB)

      if results.multi_hand_landmarks != None:
        for class_data in results.multi_handedness:
          hand_type = class_data.classification[0].label
          hand_score = round((class_data.classification[0].score * 100),2)
          hands_type.append(hand_type)
          hands_score.append(hand_score)

        for hand_landmarks in results.multi_hand_landmarks:
          my_hand = []
          for landmarks in hand_landmarks.landmark:
            my_hand.append((round(landmarks.x,2),round(landmarks.y,2),round(landmarks.z,2)))
          my_hands.append(my_hand)

          if show_detect:
            if hand_connection:
              if detection_style == 0:
                draw_landmarks(
                  image,
                  hand_landmarks,
                  hands.HAND_CONNECTIONS,
                  DrawingSpec(RED),
                  DrawingSpec(GREEN)
                )
              elif detection_style == 1:
                draw_landmarks(
                  image,
                  hand_landmarks,
                  hands.HAND_CONNECTIONS,
                  get_default_hand_landmarks_style(),
                  get_default_hand_connections_style()
                )
              else:
                raise TypeError("TypeError : detection style must have integer between 0 and 1.")
            else:
              if detection_style == 0:
                draw_landmarks(
                  image,
                  hand_landmarks,
                  None,
                  DrawingSpec(RED),
                  None
                )
              elif detection_style == 1:
                draw_landmarks(
                  image,
                  hand_landmarks,
                  None,
                  get_default_hand_landmarks_style(),
                  None
                )
              else:
                raise TypeError("TypeError : detection style must have integer between 0 and 1.")
      return my_hands,hands_type,hands_score

    except Exception as e:
      print(e)
      return None

  def find_face(self,image,mode:str = "BGR",show_detect:bool = True,boundary:bool = True):
    """It finds your face and shows face landmarks.
    
    Parameter :
    ==========
    ``image`` : BGR Image which you want to find the hands it.\n
    ``mode`` = In which image type format you have.
    Set ``BGR`` for BGR type image, set ``RGB`` for RGB type image and ``B&W`` for gray Scale image.
    Default is ``BGR``.\n
    ``show_detect`` = Set ``True`` if visually show the detection else set it ``False``. Default set it ``True``.\n
    ``boundary`` = Set ``True`` if want to set a rectangel over face else set id ``False``. Default set it ``True``.
    
    Return :
    ========
    It retuns a tuple containing all face landmarks, boundary box landmarks and detection confidence in percent else return None."""
    try:
      faces_score = []
      faces_boundry = []
      my_faces = []

      if mode == "BGR":
        image_RGB = cvtColor(image,COLOR_BGR2RGB)
      if mode == "RGB":
        image_RGB = image
      if mode == "B&W":
        image_RGB = cvtColor(image,COLOR_GRAY2RGB)

      results = self.mp_face.process(image_RGB)

      if results.detections != None:
        my_face = []
        for detection in results.detections:
          face_score = round(detection.score[0] * 100,2)
          faces_score.append(face_score)

          bbox = detection.location_data.relative_bounding_box
          faces_boundry.append((round(bbox.xmin,2),round(bbox.ymin,2),round(bbox.width,2),round(bbox.height,2)))

          for my_face_lm in detection.location_data.relative_keypoints:
            
            my_face.append((round(my_face_lm.x,2),round(my_face_lm.y,2)))
          my_faces.append(my_face)

        if show_detect:
          if boundary:
            draw_detection(
              image,
              detection,
              DrawingSpec(RED),
              DrawingSpec(GREEN)
            )
          else:
            draw_detection(
              image,
              detection,
              DrawingSpec(RED),
              None
            )
      return my_faces,faces_boundry,faces_score

    except Exception as e:
      print(e)
      return None

  def find_face_mesh(self,image,mode:str = "BGR",face_connection:bool = False,face_connection_3d:bool = False,show_detect:bool = True):
    """It finds your face and shows full face landmarks.
    
    Parameter :
    ==========
    ``image`` = Image which you want to find the hands it.\n
    ``mode`` = In which image type format you have.
    Set ``BGR`` for BGR type image, set ``RGB`` for RGB type image and ``B&W`` for gray Scale image.
    Default is ``BGR``.\n
    ``face_connection`` = Set ``True`` if want to connect all face landmarks else set ``False``. Default is ``False``.\n
    ``show_detect`` = Set ``True`` if visually show the detection else set it ``False``. Default set it ``True``.
    
    Return :
    ========
    It retuns all face landmarks  with index else for any exception it return empty string."""
    try:
      my_face_meshs = []
      
      if mode == "BGR":
        image_RGB = cvtColor(image,COLOR_BGR2RGB)
      if mode == "RGB":
        image_RGB = image
      if mode == "B&W":
        image_RGB = cvtColor(image,COLOR_GRAY2RGB)

      results = self.mp_mesh.process(image_RGB)

      if results.multi_face_landmarks != None:
        for face_landmarks in results.multi_face_landmarks:
          for face_landmark in face_landmarks.landmark:
            my_face_meshs.append((round(face_landmark.x,2),round(face_landmark.y,2),round(face_landmark.z,2)))

          if show_detect:
            if face_connection:
              if face_connection_3d:
                draw_landmarks(
                  image,
                  face_landmarks,
                  face_mesh.FACEMESH_TESSELATION,
                  None,
                  DrawingSpec(GREEN,1)
                )
              else:
                draw_landmarks(image,face_landmarks,face_mesh.FACEMESH_FACE_OVAL,None,DrawingSpec(WHITE,2))
                draw_landmarks(image,face_landmarks,face_mesh.FACEMESH_LEFT_EYE,None,DrawingSpec(GREEN,1))
                draw_landmarks(image,face_landmarks,face_mesh.FACEMESH_RIGHT_EYE,None,DrawingSpec(GREEN,1))
                draw_landmarks(image,face_landmarks,face_mesh.FACEMESH_RIGHT_EYEBROW,None,DrawingSpec(YELLOW,2))
                draw_landmarks(image,face_landmarks,face_mesh.FACEMESH_LEFT_EYEBROW,None,DrawingSpec(YELLOW,2))
                draw_landmarks(image,face_landmarks,face_mesh.FACEMESH_LIPS,None,DrawingSpec(CYAN,2))
                draw_landmarks(image,face_landmarks,face_mesh.FACEMESH_LEFT_IRIS,None,DrawingSpec(RED,2))
                draw_landmarks(image,face_landmarks,face_mesh.FACEMESH_RIGHT_IRIS,None,DrawingSpec(RED,2))
            else:
              draw_landmarks(
                image,
                face_landmarks,
                None,
                DrawingSpec(RED,1),
                None
              )

      return my_face_meshs

    except Exception as e:
      print(e)
      return None

  def find_pose(self,image,mode:str = "BGR",body_connection:bool = False,show_detect:bool = True,detection_style:int = 0):
    """It finds your full body pose and shows body landmarks.
    
    Parameter :
    ==========
    ``image`` = Image which you want to find the hands it.\n
    ``mode`` = In which image type format you have.
    Set ``BGR`` for BGR type image, set ``RGB`` for RGB type image and ``B&W`` for gray Scale image.
    Default is ``BGR``.\n
    ``body_connection`` = Set ``True`` if want to connect all body landmarks else set ``False``. Default is ``False``.\n
    ``show_detect`` = Set ``True`` if visually show the detection else set it ``False``. Default set it ``True``.\n
    ``detection_style`` = Set ``0`` for the custom drawing style or set ``1`` for the default mediapipe detection style.
    
    Return :
    ========
    It retuns a list of all body landmarks."""
    try:
      my_poses = []
      
      if mode == "BGR":
        image_RGB = cvtColor(image,COLOR_BGR2RGB)
      if mode == "RGB":
        image_RGB = image
      if mode == "B&W":
        image_RGB = cvtColor(image,COLOR_GRAY2RGB)

      results = self.mp_pose.process(image_RGB)

      if results.pose_landmarks != None:
        poses_landmark = results.pose_landmarks.landmark

        for pose_landmark in poses_landmark:
          my_poses.append((round(pose_landmark.x,2),round(pose_landmark.y,2),round(pose_landmark.z,2)))

        if show_detect:
          if body_connection:
            if detection_style == 0:
              draw_landmarks(
                image,
                results.pose_landmarks,
                pose.POSE_CONNECTIONS,
                DrawingSpec(RED),
                DrawingSpec(GREEN)
              )
            elif detection_style == 1:
              draw_landmarks(
                image,
                results.pose_landmarks,
                pose.POSE_CONNECTIONS,
                get_default_pose_landmarks_style()
              )
            else:
              raise TypeError("TypeError : detection style must have integer between 0 and 1.")

          else:
            if detection_style == 0:
              draw_landmarks(
                image,
                results.pose_landmarks,
                None,
                DrawingSpec(RED),
                None
              )
            elif detection_style == 1:
              draw_landmarks(
                image,
                results.pose_landmarks,
                None,
                get_default_pose_landmarks_style(),
                None
              )
            else:
              raise TypeError("TypeError : detection style must have integer between 0 and 1.")

      return my_poses

    except Exception as e:
      print(e)
      return None

if __name__ == '__main__':
  # FOR EXAMPLE AND DEMO RUN EACH
  from time import time
  start_time = time()
  from cv2 import imshow,waitKey,destroyAllWindows

  obj = Mptools(
    image_mode=False,
    cam_index=0,
    win_width=640,
    win_height=360,
    cam_fps=30,
    hand_no=2,
    face_no=1,
    tol1=0.5,
    tol2=0.5
  )
  cam = obj.init()

  while cam.isOpened():
    success,image = cam.read()
    if not success:
      print("\nIgnoring the empty frame...\n")
      continue

    # hand = obj.find_Hands(image=image,mode="BGR",hand_connection=True,show_detect=True,detection_style=1)
    # face = obj.find_face(image=image,mode="BGR",show_detect=True,boundary=True)
    # face_meshs = obj.find_face_mesh(image=image,mode="BGR",face_connection=True,face_connection_3d=True,show_detect=True)
    # poses = obj.find_pose(image=image,mode="BGR",body_connection=True,show_detect=True,detection_style=0)

    # if hand != ([],[],[]):
    #   print(hand)

    # if face != ([],[],[]):
    #   print(face)

    # if face_meshs != []:
    #   print(face_meshs)

    # if poses != []:
    #   print(poses)

    end_time = time()
    fps = int(1 / (end_time - start_time))
    start_time = end_time

    obj.show_FPS(image=image,mode="BGR",fps_rate=fps,fore_bg=YELLOW,back_bg=RED)

    imshow("MPKIT-CC(0.0.5) PACKAGE EXAMPLE",image)

    if waitKey(1) == ord("q"):
      break

  cam.release()
  destroyAllWindows()
