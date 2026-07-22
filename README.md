# Face-liveliness-detection

A CNN-based face anti-spoofing classifier that determines whether a face captured
by a camera is a **live person** or a **spoof** (printed photo, phone/screen replay).
Built with transfer learning (MobileNetV2) and OpenCV-based face preprocessing,
with a Streamlit demo app for interactive testing.

This project explores the kind of problem used in real-world identity verification
and facial authentication systems — detecting whether the face presented to a
camera is genuinely live before trusting it for authentication.

## How it works

1. **Face detection & cropping** (`src/preprocess.py`) — uses OpenCV's Haar Cascade
   detector to find and crop the face region from raw images, so the model only
   sees the relevant part of the frame.
2. **Classification** (`src/model.py`) — a MobileNetV2 backbone pretrained on
   ImageNet, with the convolutional layers frozen and a small custom classifier
   head trained on top (transfer learning). This keeps training fast and works
   well with a moderately sized dataset.
3. **Training & evaluation** (`src/train.py`) — trains the head, evaluates on a
   held-out validation set every epoch (accuracy, precision, recall, F1), and
   saves the best-performing checkpoint.
4. **Inference** (`src/predict.py`, `app/app.py`) — takes a new image, detects the
   face, and classifies it as Live or Spoof with a confidence score. The Streamlit
   app wraps this in a simple upload-and-classify UI.

## Project structure

```
face-liveness-detection/
├── src/
│   ├── preprocess.py    # face detection + cropping
│   ├── dataset.py       # PyTorch Dataset / DataLoader
│   ├── model.py          # MobileNetV2-based classifier
│   ├── train.py           # training loop + metrics
│   └── predict.py         # single-image inference (CLI)
├── app/
│   └── app.py              # Streamlit demo
├── data/
│   ├── real/                 # live face crops (not committed — see below)
│   └── spoof/                # spoof face crops (not committed — see below)
├── models/                     # saved model weights (not committed)
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone https://github.com/<your-username>/face-liveness-detection.git
cd face-liveness-detection
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Dataset

This repo doesn't include image data (too large for git, and not mine to
redistribute). To train the model yourself, download one of these public
anti-spoofing datasets and place the images into `raw_data/real/` and
`raw_data/spoof/` before preprocessing:

- **NUAA Photograph Imposter Database** — a well-known, lightweight face
  anti-spoofing dataset, good for a first working version.
- **CelebA-Spoof** — a much larger dataset (~600K images) with more spoof
  variety (print, replay, 3D mask), if you want stronger results.
Then run preprocessing to detect and crop faces into `data/`:

```bash
python src/preprocess.py --input_dir raw_data/real  --output_dir data/real
python src/preprocess.py --input_dir raw_data/spoof --output_dir data/spoof
```

## Training

```bash
python src/train.py --data_dir data --epochs 15 --batch_size 32
```

This prints accuracy, precision, recall, and F1 on the validation set each
epoch, saves the best checkpoint to `models/liveness_model.pt`, and reports
final metrics on the held-out test set.

**Results (fill in after training on your machine):**

| Metric    | Value |
|-----------|-------|
| Accuracy  | TBD   |
| Precision | TBD   |
| Recall    | TBD   |
| F1        | TBD   |

## Running inference

Single image via CLI:

```bash
python src/predict.py --image path/to/photo.jpg
```

Interactive demo:

```bash
streamlit run app/app.py
```

## Possible extensions

- Swap the Haar Cascade detector for MTCNN or a DNN-based detector for more
  robust face localization
- Add temporal/video-based liveness cues (blink detection, head movement)
  instead of single-frame classification
- Fine-tune deeper layers of the backbone (`unfreeze_backbone()` in
  `model.py` is already set up for this) once the head has converged
- Export the model with ONNX or TorchScript for faster inference

## Tech stack

Python, PyTorch, torchvision, OpenCV, scikit-learn, Streamlit

