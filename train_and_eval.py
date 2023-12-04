import sys
import torch
import torcheval.metrics.functional as metric
from constants import (
    CHANDLER_DETECTOR_PATH,
    DEVICE,
    FRIENDS_CLASSIFIER_PATH,
    LABELS,
    LABELS_MAP,
)
from dataset import FriendsDataset
from model import BinaryClassifier, MulticlassClassifier, BaselineModel


class ModelTrainer:
    """Trainer class for PyTorch models.

    This class handles the training loop for a given PyTorch model including
    loss computation, backpropagation, and statistics tracking.
    """

    def __init__(self, name, model, loss_fn, optimizer):
        self.name = name
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.batch_num = 0
        self.total_loss = 0

    def train(self, inputs: torch.Tensor, labels: torch.Tensor) -> float:
        """Trains the model on a batch of data.

        Args:
            inputs: Input data batch.
            labels: Corresponding labels for the input data.

        Returns:
            Loss for this batch of data.
        """
        self.model.train(True)
        self.optimizer.zero_grad()
        outputs = self.model(inputs.detach())

        # Compute the loss and its gradients.
        loss = self.loss_fn(outputs, labels.detach())
        loss.backward()

        # Adjust learning weights.
        self.optimizer.step()

        # Update stats.
        self.batch_num += 1
        self.total_loss += float(loss.item())

        return loss.item()

    def __str__(self):
        batch_loss = (
            (self.total_loss / self.batch_num) if self.batch_num > 0 else 0
        )
        return (
            f"{self.name}: {self.total_loss:.5f} "
            f"({batch_loss:.5f} per batch)"
        )

    def reset_stats(self):
        """Resets the batch statistics."""
        self.batch_num = 0
        self.total_loss = 0


if __name__ == "__main__":
    print(f"Using device: {DEVICE}.")

    trainset, testset = torch.utils.data.random_split(
        FriendsDataset(img_dir="data", cache_all=False),
        (0.8, 0.2),
        generator=torch.Generator().manual_seed(0),
    )

    # Training starts
    if "--train" in sys.argv or "-t" in sys.argv:
        training_loader = torch.utils.data.DataLoader(
            trainset, batch_size=64, shuffle=True
        )

        friends_classifier = MulticlassClassifier()
        chandler_detector = BinaryClassifier()
        friends_classifier.to(DEVICE)
        chandler_detector.to(DEVICE)

        friends_classifier_trainer = ModelTrainer(
            "friends_classifier",
            friends_classifier,
            torch.nn.CrossEntropyLoss(),
            torch.optim.Adam(friends_classifier.parameters(), lr=0.0005),
        )
        chandler_detector_trainer = ModelTrainer(
            "chandler_detector",
            chandler_detector,
            torch.nn.BCELoss(),
            torch.optim.Adam(chandler_detector.parameters(), lr=0.0005),
        )

        for epoch in range(1000):
            print(f"Epoch: {epoch}")
            for i, (inputs, labels, _) in enumerate(training_loader):
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
                friends_classifier_trainer.train(inputs, labels)

                is_chandler = labels.float()
                is_chandler[labels != 1] = 0
                chandler_detector_trainer.train(inputs, is_chandler)

            print(friends_classifier_trainer)
            print(chandler_detector_trainer)
            friends_classifier_trainer.reset_stats()
            chandler_detector_trainer.reset_stats()

        torch.save(friends_classifier, FRIENDS_CLASSIFIER_PATH)
        torch.save(chandler_detector, CHANDLER_DETECTOR_PATH)
    # Training ends

    # Validation starts
    baseline = BaselineModel()
    friends_classifier = torch.load(
        FRIENDS_CLASSIFIER_PATH, map_location=DEVICE
    ).eval()
    chandler_detector = torch.load(
        CHANDLER_DETECTOR_PATH, map_location=DEVICE
    ).eval()
    friends_classifier.to(DEVICE)
    chandler_detector.to(DEVICE)

    testing_loader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset), shuffle=True
    )
    with torch.no_grad():  # No gradients needed for validation
        inputs, labels, images = next(iter(testing_loader))
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        pred_baseline = (
            torch.Tensor([LABELS_MAP[p] for p in baseline.predict(inputs)])
            .long()
            .to(DEVICE)
        )
        pred_friends_classifier = friends_classifier(inputs)

        print("------------------------------")
        print("Multiclass performance:")
        accuracy = lambda x, a="micro": metric.multiclass_accuracy(
            x, labels, average=a, num_classes=len(LABELS)
        )
        precision = lambda x, a="micro": metric.multiclass_precision(
            x, labels, average=a, num_classes=len(LABELS)
        )
        recall = lambda x, a="micro": metric.multiclass_recall(
            x, labels, average=a, num_classes=len(LABELS)
        )
        f1_score = lambda x, a="micro": metric.multiclass_f1_score(
            x, labels, average=a, num_classes=len(LABELS)
        )

        for name, pred in (
            ("baseline", pred_baseline),
            ("friends_classifier", pred_friends_classifier),
        ):
            print(f"\n# {name} model:")
            print("|  class   | accuracy | precision | recall  | f1_score |")
            print("|----------|----------|-----------|---------|----------|")
            print(
                "| average  "
                + "| {:<8} ".format(f"{accuracy(pred):.2%}")
                + "| {:<9} ".format(f"{precision(pred):.2%}")
                + "| {:<7} ".format(f"{recall(pred):.2%}")
                + "| {:<8} |".format(f"{f1_score(pred):.2%}")
            )
            for i, (a, p, r, f) in enumerate(
                zip(
                    accuracy(pred, None),
                    precision(pred, None),
                    recall(pred, None),
                    f1_score(pred, None),
                )
            ):
                print(
                    f"| {LABELS[i]:<8} "
                    + "| {:<8} ".format(f"{a:.2%}")
                    + "| {:<9} ".format(f"{p:.2%}")
                    + "| {:<7} ".format(f"{r:.2%}")
                    + "| {:<8} |".format(f"{f:.2%}")
                )

        print("------------------------------")
        print("Binary performance:")
        labels[labels != 1] = 0
        pred_baseline[pred_baseline != 1] = 0
        pred_friends_classifier = pred_friends_classifier.argmax(dim=-1)
        pred_friends_classifier[pred_friends_classifier != 1] = 0
        pred_chandler_detector = chandler_detector(inputs)

        accuracy = lambda x: metric.binary_accuracy(x, labels)
        precision = lambda x: metric.binary_precision(x, labels)
        recall = lambda x: metric.binary_recall(x, labels)
        f1_score = lambda x: metric.binary_f1_score(x, labels)

        print(
            "|        model       | accuracy "
            "| precision | recall  | f1_score |"
        )
        print(
            "|--------------------|----------"
            "|-----------|---------|----------|"
        )
        for name, pred in (
            ("baseline", pred_baseline),
            ("friends_classifier", pred_friends_classifier),
            ("chandler_detector", pred_chandler_detector),
        ):
            print(
                f"| {name:<18} "
                + "| {:<8} ".format(f"{accuracy(pred):.2%}")
                + "| {:<9} ".format(f"{precision(pred):.2%}")
                + "| {:<7} ".format(f"{recall(pred):.2%}")
                + "| {:<8} |".format(f"{f1_score(pred):.2%}")
            )
    # Validation ends
