# LSTM Emotion Classification

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

This repository provides an implementation of emotion classification using Long Short-Term Memory (LSTM) networks. The project aims to classify human emotions from text data leveraging deep learning models, primarily built using Python and Jupyter Notebooks.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Language Composition](#language-composition)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Features

- LSTM-based model for emotion classification
- Written primarily in Jupyter Notebook for interactive analysis and experimentation
- Data preprocessing and exploratory data analysis
- Model training, evaluation, and prediction
- Visualization of results

## Language Composition

- **Jupyter Notebook:** 93.6%
- **Python:** 6.4%

## Project Structure

```
.
├── data/                   # Training and testing datasets (not included due to size/licensing)
├── notebooks/              # Jupyter Notebooks for data exploration, training, and evaluation
│   └── LSTM_Emotion_Classification.ipynb
├── src/                    # Python scripts for preprocessing, model, and utility functions
├── requirements.txt        # Dependencies for the project
├── README.md               # Project documentation (this file)
└── LICENSE                 # License information
```

## Installation

1. **Clone this repository:**
    ```bash
    git clone https://github.com/Subhajitde50000/LSTM_emotion_classification.git
    cd LSTM_emotion_classification
    ```

2. **Create a virtual environment (optional but recommended):**
    ```bash
    python -m venv env
    source env/bin/activate  # On Windows: env\Scripts\activate
    ```

3. **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Place your dataset in the `data/` directory. The dataset should contain text samples and their corresponding emotion labels.

2. Open the main Jupyter Notebook:
    ```bash
    jupyter notebook notebooks/LSTM_Emotion_Classification.ipynb
    ```
3. Follow the notebook sections for data preprocessing, training, and evaluation.

4. For reusing the model or running predictions on new data, use the provided utility scripts in the `src/` directory or extend with your own scripts.

## Dataset

- **Note:** The dataset used for training and testing is not included in this repository due to size or licensing restrictions. You may use publicly available datasets such as:
    - [Emotion Dataset](https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp) from Kaggle
    - Or your own preprocessed dataset

- Ensure the dataset is organized with at least two columns:
    - `text`: The text sample (sentence, paragraph, or utterance)
    - `label`: The corresponding emotion (e.g., happy, sad, angry, etc.)

## Results

After training, the notebook provides:
- Training and validation accuracy/loss plots
- Classification report (precision, recall, F1-score)
- Confusion matrix for performance visualization

*(Insert example results or output images here if available.)*

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements, new features, or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

**Contact:**  
For any questions or suggestions, feel free to open an issue or contact the repository owner.
