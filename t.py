import face_recognition
import os
import cv2
import numpy as np
from numpy import *
import dlib
import face_recognition_models
from scipy.spatial import distance as dist

def eye_aspect_ratio(eye):
    # compute the euclidean distance between the vertical eye landmarks
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal eye landmarks
    C = dist.euclidean(eye[0], eye[3])

    # compute the EAR
    ear = (A + B) / (2 * C)
    return ear

def face_encodings1(face_image, known_face_locations=None, num_jitters=1):
    """
    Given an image, return the 128-dimension face encoding for each face in the image.

    :param face_image: The image that contains one or more faces
    :param known_face_locations: Optional - the bounding boxes of each face if you already know them.
    :param num_jitters: How many times to re-sample the face when calculating encoding. Higher is more accurate, but slower (i.e. 100 is 100x slower)
    :return: A list of 128-dimensional face encodings (one for each face in the image)
    """
    raw_landmarks = face_recognition.api._raw_face_landmarks(face_image, known_face_locations, model="large")
    return [np.array(face_encoder.compute_face_descriptor(face_image, raw_landmark_set, num_jitters)) for raw_landmark_set in raw_landmarks]

def face_distance(face_encodings, face_to_compare):
    """
    Given a list of face encodings, compare them to a known face encoding and get a euclidean distance
    for each comparison face. The distance tells you how similar the faces are.

    :param faces: List of face encodings to compare
    :param face_to_compare: A face encoding to compare against
    :return: A numpy ndarray with the distance for each face in the same order as the 'faces' array
    """
    if len(face_encodings) == 0:
        return np.empty((0))
    return np.linalg.norm(face_encodings - face_to_compare, axis=1)

def read_file(path):
    face_encoding_list = []
    label_list = []
    dir_counter = 0

    #对路径下的所有子文件夹中的所有jpg文件进行读取并存入到一个list中
    for child_dir in os.listdir(path):
         child_path = os.path.join(path, child_dir)

         for dir_image in  os.listdir(child_path):
             if endwith(dir_image,'jpg'):
                img = face_recognition.load_image_file(os.path.join(child_path, dir_image))
                face_encoding=face_encodings1(img)[0]
                face_encoding_list.append(face_encoding)
                label_list.append(dir_counter)
         dir_counter += 1
    return face_encoding_list,label_list,dir_counter

#输入一个字符串一个标签，对这个字符串的后续和标签进行匹配
def endwith(s,*endstring):
   resultArray = map(s.endswith,endstring)
   if True in resultArray:
       return True
   else:
       return False

#读取训练数据集的文件夹，把他们的名字返回给一个list
def read_name_list(path):
    name_list = []
    for child_dir in os.listdir(path):
        name_list.append(child_dir)
    return name_list

face_recognition_model = face_recognition_models.face_recognition_model_location()
face_encoder = dlib.face_recognition_model_v1(face_recognition_model)
video_capture = cv2.VideoCapture(0)
EYE_AR_THRESH = 0.28
EYE_AR_CONSEC_FRAMES = 3
EAR_AVG = 0

COUNTER = 0
TOTAL = 0

if __name__ == '__main__':
    known_face_encodings, label_lsit, counter = read_file('data')
    # print(known_face_encodings)
    # print(label_lsit)
    # print(len(label_lsit))
    # print(counter)
    # 读取data数据集下的子文件夹名称
    known_face_names = read_name_list('data')
    # print(known_face_names)
    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_frame = frame[:, :, ::-1]

        # Find all the faces and face enqcodings in the frame of video
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_encodings1(rgb_frame, face_locations)
        face_landmarks = face_recognition.face_landmarks(rgb_frame, face_locations)

        # Loop through each face in this frame of video
        for (top, right, bottom, left), face_encoding,face_landmark in zip(face_locations, face_encodings,face_landmarks):
            # See if the face is a match for the known face(s)
            # matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            # matches=list(face_distance(known_face_encodings, face_encoding) <= 0.4)
            distance=list(face_distance(known_face_encodings, face_encoding))
            name = "Stranger"
            dmin=min(distance)
            if dmin<=0.4:
                index=distance.index(dmin)

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:

                # first_match_index = matches.index(True)
                name = known_face_names[index]
                # print(face_landmark['left_eye'])

                left_eye =mat(face_landmark['left_eye'])
                right_eye =mat(face_landmark['right_eye'])
                # print(left_eye)
                # draw contours on the eyes
                left_eye_hull = cv2.convexHull(left_eye)
                right_eye_hull = cv2.convexHull(right_eye)
                cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0),
                                 1)  # (image, [contour], all_contours, color, thickness)
                cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)
                # # compute the EAR for the left eye
                ear_left = eye_aspect_ratio(left_eye)
                # compute the EAR for the right eye
                ear_right = eye_aspect_ratio(right_eye)
                # compute the average EAR
                ear_avg = (ear_left + ear_right) / 2.0
                # detect the eye blink
                if ear_avg < EYE_AR_THRESH:
                    COUNTER += 1
                else:
                    if COUNTER >= EYE_AR_CONSEC_FRAMES:
                        TOTAL += 1
                        print("Eye blinked")
                    COUNTER = 0

                cv2.putText(frame, "Blinks{}".format(TOTAL), (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 255), 1)
                # cv2.putText(frame, "EAR {}".format(ear_avg), (10, 60), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 255), 1)

                # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left,bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Display the resulting image
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()