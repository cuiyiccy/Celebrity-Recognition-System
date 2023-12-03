import torch
from facenet_pytorch import InceptionResnetV1, MTCNN

# The default input image size for the models.
INPUT_IMAGE_SIZE = 160

# Constants representing labels for the dataset.
# These labels correspond to character names from the TV show "Friends".
LABELS = [
    "unknown",
    "chandler",
    "joey",
    "monica",
    "phoebe",
    "rachel",
    "ross",
]
LABELS_MAP = {
    "unknown": 0,
    "chandler": 1,
    "joey": 2,
    "monica": 3,
    "phoebe": 4,
    "rachel": 5,
    "ross": 6,
}

# Determine the best available device for tensor computations.
DEVICE = (
    torch.device("cuda")
    if torch.cuda.is_available()
    # mps is bugged
    # else torch.device("mps")
    # if torch.backends.mps.is_available()
    else torch.device("cpu")
)

# Initialize a shared MTCNN instance for face detection.
# MTCNN is used to detect and align faces in the input images.
SHARED_MTCNN = MTCNN(
    image_size=INPUT_IMAGE_SIZE,
    margin=0,
    min_face_size=20,
    thresholds=[0.7, 0.8, 0.8],
    factor=0.709,
    post_process=True,
    keep_all=True,
    device=DEVICE,
).eval()

# Initialize a shared Inception Resnet V1 instance for facial recognition.
# This model is pretrained on the 'vggface2' dataset.
SHARE_RESNET = InceptionResnetV1(
    pretrained="vggface2",
    device=DEVICE,
).eval()

# Default paths to saved models.
FRIENDS_CLASSIFIER_PATH = "saved_models/friends_classifier"
CHANDLER_DETECTOR_PATH = "saved_models/chandler_detector"
