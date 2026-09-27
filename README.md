# Intel Scene Classification with a Residual CNN

A TensorFlow image-classification project that trains a custom residual convolutional neural network on the [Intel Image Classification dataset](https://www.kaggle.com/datasets/puneet6060/intel-image-classification). The notebook covers the complete workflow: reproducible data ingestion, augmentation, model design, controlled training, evaluation, and Grad-CAM interpretation.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YousefAhmed24/Intel-CNN/blob/main/Deep_Learning.ipynb)

## Results

The model was evaluated on a stratified test set of 2,556 images.

| Metric | Result |
|---|---:|
| Test accuracy | 80.01% |
| Macro F1-score | 0.80 |
| Weighted F1-score | 0.80 |

The strongest class result was for `forest` (0.93 F1). The most frequent confusion was `buildings` classified as `street`, with a normalized confusion rate of 0.34.

## What the notebook implements

- Programmatic dataset download with `kagglehub`
- Reproducible 70/15/15 stratified train, validation, and test split
- Efficient `tf.data` pipelines with caching, parallel mapping, batching, and prefetching
- Random flip, rotation, and zoom augmentation
- A custom Keras Functional API CNN with batch normalization, spatial dropout, and a residual connection
- AdamW optimization with learning-rate reduction and early stopping
- Per-class precision, recall, and F1 evaluation
- Normalized confusion-matrix analysis
- Grad-CAM visualizations for correct and incorrect predictions

## Model architecture

The network accepts 150 × 150 RGB images and uses:

1. A 32-filter convolutional stem and max pooling
2. A 64-filter feature-extraction block with spatial dropout
3. A two-layer 64-filter residual block with an identity skip connection
4. A 128-filter convolutional block
5. Global average pooling and a six-class softmax output

## Dataset

The notebook downloads the dataset automatically and excludes the unlabeled `seg_pred` directory. The six classes are:

- `buildings`
- `forest`
- `glacier`
- `mountain`
- `sea`
- `street`

The run recorded in the notebook uses 17,034 labeled images: 11,923 for training, 2,555 for validation, and 2,556 for testing.

## Run the project

The easiest option is to open the notebook in Google Colab and select a GPU runtime.

For a local Jupyter environment, install the main dependencies:

```bash
pip install tensorflow scikit-learn matplotlib seaborn opencv-python kagglehub numpy
```

Then open and run `Deep_Learning.ipynb` from top to bottom. `kagglehub` may ask you to authenticate before downloading the dataset.

## Repository contents

```text
Deep_Learning.ipynb  End-to-end training, evaluation, and interpretation notebook
README.md            Project overview and reproduction notes
```

## Notes

The reported metrics are from the saved notebook run and may vary slightly across TensorFlow versions and hardware. This is an educational computer-vision project, not a production classifier.