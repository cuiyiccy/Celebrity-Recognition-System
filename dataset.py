"""Module for loading and processing the Friends image dataset.

This module includes functionality to load images from a specified directory,
preprocess them using the MTCNN and InceptionResnetV1 models from
facenet_pytorch, and create a dataset compatible with PyTorch's `DataLoader`.
"""

import os
from typing import Any, Callable, List, Tuple, Optional

import cv2
import torch
from constants import (
    LABELS,
    LABELS_MAP,
    DEVICE,
    SHARED_MTCNN,
    SHARE_RESNET,
    INPUT_IMAGE_SIZE,
)


def load_and_preprocess_image(img_path: str) -> Optional[torch.Tensor]:
    """
    Loads and preprocesses an image for facial recognition.

    The function reads an image from a given path, then uses the shared MTCNN
    model for face detection and preprocessing. If face detection fails, it
    manually resizes and normalizes the image. The function also handles cases
    where multiple faces are detected by returning None.

    Args:
        img_path: The file path of the image to be processed.

    Returns:
        A tensor representing the preprocessed image, or None if multiple faces
    are detected.
    """
    img = cv2.imread(img_path)
    preprocessed_img = SHARED_MTCNN(torch.Tensor(img).to(DEVICE))

    # Manually process the image if MTCNN detection failed.
    if preprocessed_img is None:
        # print(f"Face detection failed, using raw image as-is: {img_path}")
        img = cv2.resize(img, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE))
        preprocessed_img = torch.Tensor(img.transpose(2, 0, 1)).to(DEVICE)
        preprocessed_img = preprocessed_img.unsqueeze(dim=0)  # Add batch dim
        if SHARED_MTCNN.post_process:
            # Nomalize intensity.
            preprocessed_img = (preprocessed_img - 127.5) / 128.0
    # Bad example: multiple faces detected.
    elif preprocessed_img.size()[0] != 1:
        print(f"Detected multiple faces, skipping bad example: {img_path}")
        return None

    return preprocessed_img


class FriendsDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset class for loading and transforming images from a specified
    directory, using the directory name as the label.
    """

    def __init__(
        self,
        img_dir: str,
        loader_func: Callable[
            [Any], Optional[torch.Tensor]
        ] = load_and_preprocess_image,
        encode_func: Callable[[torch.Tensor], torch.Tensor] = SHARE_RESNET,
    ) -> None:
        self.labels: List[int] = []
        self.images: List[str] = []
        self.embeddings: List[torch.Tensor] = []
        for d in os.listdir(img_dir):
            sub_dir = os.path.join(img_dir, d)
            if d not in LABELS or not os.path.isdir(sub_dir):
                continue
            for i in os.listdir(sub_dir):
                if i.split(".")[-1] not in ("jpg", "jpeg", "png"):
                    continue
                img_path = os.path.join(sub_dir, i)
                img = loader_func(img_path)
                if img is None:
                    continue
                self.labels.append(LABELS_MAP[d])
                self.images.append(os.path.join(sub_dir, i))
                self.embeddings.append(encode_func(img).squeeze())

    def __len__(self) -> int:
        """Returns the number of items in the dataset."""
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        """
        Retrieves an item from the dataset by index.

        Args:
            idx: The index of the item.

        Returns:
            A tuple containing the image tensor, its corresponding label, and
        the path of the raw image.
        """
        return self.embeddings[idx], self.labels[idx], self.images[idx]


# Test only
if __name__ == "__main__":
    loader = torch.utils.data.DataLoader(
        FriendsDataset(img_dir="samples"), batch_size=100, shuffle=True
    )
    for i, (inputs, labels, images) in enumerate(loader):
        print(inputs.size())
        print(labels.size())
        print(images)
        break
