# Friends Celebrity Recognition

## Overview

This repository contains the implementation for a facial recognition system designed to identify characters from the TV show "Friends". It includes a training script for both binary and multiclass classifiers, a baseline model using nearest neighbor, and an application for tagging detected faces in videos.

## Directory Structure

- `data/`: Contains cropped images of "Friends" characters for training, validation, and testing. Subfolder names are used as labels.
- `results/`: Result video with annotation of celebrity.
- `samples/`: Example images for the baseline nearest neighbor model.
- `saved_models/`: Stores the trained models.
- `app.py`: Main entry point for running face tagging on videos.
- `train_and_eval.py`: Main entry point for training and evaluating the models.
- `project_presentation.mp4`: A video presentation to introduce the model and experience.
- `Final Report - Celebrity Recognition.pdf`: Detailed report about the project.

## Prerequisites

- Python 3.8 or higher
- pipenv

## Setup

1. Clone the repository and navigate to its root directory.
2. Run `pipenv install` to set up the environment.
  - To use GPUs, you may need to manually install `pytorch` by following [this instruction](https://pytorch.org/get-started/locally/).
3. Activate the environment with `pipenv shell`.

## Running the Application

- To run the main application on a video, use:
  ```
  python app.py [video_path]
  ```
  This will run MTCNN face detection and apply both the trained models and the baseline model to tag the detected faces.

## Training and Evaluation

- To train and evaluate the models, run:
  ```
  python train_and_eval.py -t
  ```
  or
  ```
  python train_and_eval.py --train
  ```
  This will train & save the models, and evaluate them against the baseline model.

- To skip training and only run evaluation, omit `-t` or `--train`:
  ```
  python train_and_eval.py
  ```

- If you are running into OOM issue when loading the dataset, try disabling `cache_all` in `train_and_eval.py`:

  ```
  FriendsDataset(img_dir="data", cache_all=False)
  ```

### Eval Results as of Current Commit

#### Multiclass Performance

##### Baseline Model

|  class   | accuracy | precision | recall  | f1_score |
|----------|----------|-----------|---------|----------|
| average  | 63.08%   | 63.08%    | 63.08%  | 63.08%   |
| unknown  | 100.00%  | 19.66%    | 100.00% | 32.86%   |
| chandler | 57.98%   | 100.00%   | 57.98%  | 73.40%   |
| joey     | 78.95%   | 88.24%    | 78.95%  | 83.33%   |
| monica   | 26.92%   | 100.00%   | 26.92%  | 42.42%   |
| phoebe   | 25.00%   | 100.00%   | 25.00%  | 40.00%   |
| rachel   | 58.82%   | 100.00%   | 58.82%  | 74.07%   |
| ross     | 100.00%  | 100.00%   | 100.00% | 100.00%  |

#### Custom Trained Model

|  class   | accuracy | precision | recall  | f1_score |
|----------|----------|-----------|---------|----------|
| average  | 99.23%   | 99.23%    | 99.23%  | 99.23%   |
| unknown  | 95.65%   | 100.00%   | 95.65%  | 97.78%   |
| chandler | 99.16%   | 100.00%   | 99.16%  | 99.58%   |
| joey     | 100.00%  | 100.00%   | 100.00% | 100.00%  |
| monica   | 100.00%  | 96.30%    | 100.00% | 98.11%   |
| phoebe   | 100.00%  | 100.00%   | 100.00% | 100.00%  |
| rachel   | 100.00%  | 97.14%    | 100.00% | 98.55%   |
| ross     | 100.00%  | 100.00%   | 100.00% | 100.00%  |

#### Binary (Chandler vs non-Chandler) performance:

|        model       | accuracy | precision | recall  | f1_score |
|--------------------|----------|-----------|---------|----------|
| baseline           | 80.77%   | 100.00%   | 57.98%  | 73.40%   |
| friends_classifier | 99.62%   | 100.00%   | 99.16%  | 99.58%   |
| chandler_detector  | 100.00%  | 100.00%   | 100.00% | 100.00%  |
