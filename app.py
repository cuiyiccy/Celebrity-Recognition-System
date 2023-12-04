import sys
import cv2
import torch
from constants import (
    CHANDLER_DETECTOR_PATH,
    DEVICE,
    FRIENDS_CLASSIFIER_PATH,
    SHARE_RESNET,
    SHARED_MTCNN,
)
from model import BaselineModel, MulticlassClassifier, BinaryClassifier


def annotateVideo(video_source, models, detection_threshold=0.9):
    cap = cv2.VideoCapture(video_source)

    # process a single frame of video
    while cap.isOpened():
        _, frame = cap.read()  # grab a single frame
        if frame is None:
            break

        boxes, probs = SHARED_MTCNN.detect(frame)  # type: ignore
        faces = SHARED_MTCNN.extract(frame, boxes, None)
        if faces is None:
            continue

        embeddings = SHARE_RESNET(faces.to(DEVICE))  # type: ignore
        predictions = {
            name: model.predict(embeddings) for name, model in models.items()
        }

        for i, (box, prob) in enumerate(zip(boxes, probs)):
            if prob < detection_threshold:
                continue

            # draw box, name, and prob
            x, y = int(box[0]), int(box[1])
            start_point = (x, y)
            end_point = (int(box[2]), int(box[3]))
            color = (0, 0, 255)
            cv2.rectangle(frame, start_point, end_point, color, 2)
            cv2.putText(
                frame,
                f"MTCNN: {prob:.3%}",
                (x, y + 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,  # fontScale
                (255, 255, 255),
                1,
            )
            for j, (name, outputs) in enumerate(predictions.items()):
                cv2.putText(
                    frame,
                    f"{name}: {outputs[i]}",
                    (x, y - 5 - 18 * j),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,  # fontScaleq
                    (255, 255, 255),
                    1,
                )

        # process = not process
        cv2.imshow("frame", frame)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break
        if key == ord("p"):
            cv2.waitKey(-1)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_source = sys.argv[1] if len(sys.argv) > 1 else "./videos/test00.mov"

    friends_classifier = torch.load(
        FRIENDS_CLASSIFIER_PATH, map_location=DEVICE
    ).eval()
    chandler_detector = torch.load(
        CHANDLER_DETECTOR_PATH, map_location=DEVICE
    ).eval()
    friends_classifier.to(DEVICE)
    chandler_detector.to(DEVICE)

    annotateVideo(
        video_source,
        {
            "baseline": BaselineModel(),
            "friends_classifier": friends_classifier,
            "chandler_detector": chandler_detector,
        },
    )
