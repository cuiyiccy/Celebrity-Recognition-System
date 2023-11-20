from os import listdir
from os.path import isfile, join
import cv2
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
import numpy as np

##### load sample pictures and save face encoding for each person #####


def getEnbedding(mtcnn, resnet, sample_path):

    # get list of people names
    people_names = [file for file in listdir(sample_path)]
    people_names = [file.split(".")[0] for file in people_names if file.split(".")[
        1] in ('png')]
    # a dictionary to store all the sample face embeddings
    people_faces_enbeddings = {}

    for people in people_names:
        img = cv2.imread(f'{sample_path}/{people}.png')

        resized_img = mtcnn(img)
        if resized_img is not None:
            people_faces_enbeddings[people] = resnet(
                torch.Tensor(resized_img))[0, :]
    # print(people_faces_enbedding)
    # print(people_names)
    return people_faces_enbeddings


# detect faces
# input:video_source can be either the device index(0) or the name of a video file
def detectFace(video_source, people_faces_enbeddings, thres=0.7):

    cap = cv2.VideoCapture(video_source)
    # process = True  # use for skipping every other frame for faster processing

    # process a single frame of video
    while cap.isOpened():
        _, frame = cap.read()  # grab a single frame
        # frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25) #resize to smaller frame for faster processing
        boxes, probs = mtcnn.detect(frame)
        faces = mtcnn.extract(frame, boxes, None)

        if faces is not None:
            for box, prob, face in zip(boxes, probs, faces):
                detected_enbedding = resnet(
                    torch.Tensor(face.unsqueeze(0)))[0, :]
                # find the best match face from the samples
                compare_result = {}
                for name, face in people_faces_enbeddings.items():
                    compare_result[name] = (
                        face - detected_enbedding).norm().item()

                best_match = min(compare_result, key=compare_result.get)
                # if (compare_result[best_match] >= thres):
                #     best_match = "Unknown"

                # draw box, name, and prob
                x, y = int(box[0]), int(box[1])
                start_point = (x, y)
                end_point = (int(box[2]), int(box[3]))
                color = (0, 0, 255)
                prob = round(prob * 100, 2)
                text = best_match + " " + str(prob) + "%"
                cv2.rectangle(frame, start_point, end_point, color, 2)
                cv2.putText(
                    frame, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)

        # process = not process
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # create detector
    mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20,
                  thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True, keep_all=True)
    # load model
    resnet = InceptionResnetV1(pretrained='vggface2').eval()

    sample_path = "./samples"
    video_source = "./friends.mov"

    people_faces_enbeddings = getEnbedding(mtcnn, resnet, sample_path)
    detectFace(video_source, people_faces_enbeddings, thres=0.7)
