# Computer Vision (ITE4052)

Coursework for Computer Vision, Hanyang University, 2025-Fall — three assignments
and the team term project.

```
assignments/                 course assignments built on the ITE4052 helpers
├── a1/                      kNN classifier, PyTorch tutorial
├── a2/                      linear classifier, two-layer network
└── a3/                      fully connected and convolutional networks
thermal-anomaly-detection/   team term project
```

## Term project — Thermal Anomaly Hunter

Team **VisionaryAI**. Object detection on thermal imagery: an anomaly is a
region that is hotter than its surroundings, so the task is framed as detection
rather than classification — find the blob, not just label the frame.

YOLOv8n and YOLOv8s were trained first as baselines, then YOLO11n and YOLO11s.
The final study repeats YOLO11s with three different random seeds to check how
much of the reported score is luck.

| model | seed | precision | recall | mAP@0.5:0.95 | score |
|---|---|---|---|---|---|
| YOLO11s | 42 | 0.9990 | 0.9954 | **0.9786** | 0.9823 |
| YOLO11s | 77 | 0.9958 | 0.9945 | 0.9599 | 0.9670 |
| YOLO11s | 2025 | 0.9976 | 0.9946 | 0.9578 | 0.9655 |

`score` is the weighted combination used for the internal comparison
(0.1 precision + 0.1 recall + 0.8 mAP@0.5:0.95).

Spread across seeds is about 0.02 mAP@0.5:0.95, which is the honest error bar
for any single run. Validation set: 1080 images, 1315 instances. Training ran on
an RTX A5000; the notebooks assume a folder layout of `dataset/` next to
`thermal.yaml`, and results land in `runs/detect/`.

```
thermal-anomaly-detection/
├── project.ipynb    data inspection, label conversion, training and comparison of
│                    YOLOv8n / YOLOv8s / YOLO11n / YOLO11s
└── week14.ipynb     seed-variance study: three YOLO11s runs, metrics collected
                     from each run's results.csv
```

## Assignments

| folder | what it covers |
|---|---|
| `a1/` | kNN image classifier written from scratch, plus the PyTorch tutorial notebook |
| `a2/` | linear classifiers (SVM and softmax) and a two-layer network with hand-written backward pass |
| `a3/` | fully connected and convolutional networks, including the modular layer API |

Each folder carries the `ite4052` helper package the course provides (data
loading, gradient checking, solvers). CIFAR-10 downloads on first run and is not
committed.

## Not in this repository

| excluded | why |
|---|---|
| lecture slides (`2. Image Classification.pdf` … `9. Training Neural Networks.pdf`) | course material |
| proposal report, mid-term deck, final deck, written report | presentation and submission material |
| submission PDFs and `.zip` bundles | duplicates of the assignment folders |
| datasets and trained weights | CIFAR-10 is downloaded; the thermal set and `*.pt` weights are not redistributed |

## Team

Term project by VisionaryAI.
