# Fine-Tuning Transformers for Toponym Resolution
## A Contextual Embedding Approach to Candidate Ranking
---
## Abstract
We introduce a new approach to toponym resolution, leveraging transformer-based Siamese networks to disambiguate geographical references in unstructured text. Our methodology consists of two steps: the generation of location candidates using the GeoNames gazetteer, and the ranking of these candidates based on their semantic similarity to the toponym in its document context. The core of the proposed method lies in the adaption of SentenceTransformer models, originally designed for sentence similarity tasks, to toponym resolution by fine-tuning them on geographically annotated English news article datasets (Local Global Lexicon, GeoWebNews, and TR-News). The models are used to generate contextual embeddings of both toponyms and textual representations of location candidates, which are then used to rank candidates using cosine similarity. The results suggest that the fine-tuned models outperform existing solutions in several key metrics.

## Demonstration
For a practical demonstration of our approach, we have prepared the `demo.ipynb` notebook. It provides a step-by-step overview of the entire process, from data preparation and pre-processing to model fine-tuning and evaluation.

## Setup Instructions

Follow these steps to set up your environment and run the `demo.ipynb` notebook.

### 1. Clone the Repository

Clone the repository into your desired directory:

```
git clone https://github.com/dguzh/SemTopRes.git
```

```
cd SemTopRes
```

### 2. Setting Up the Jupyter Notebook Environment

Ensure you have a Jupyter notebook environment installed. If not, you can install JupyterLab using pip:

```
pip3 install jupyterlab
```

Also make sure to have ipywidgets installed to correctly display progress bars:

```
pip3 install ipywidgets
```

### 3. Creating a Virtual Environment

Create a virtual environment in the directory:

For **Linux/macOS**:
```
python3 -m venv myenv
```

For **Windows**:
```
python -m venv myenv
```

Replace `myenv` with your preferred name for the virtual environment.

### 4. Activating the Virtual Environment

Activate the virtual environment:

On **Linux/macOS**:
```
source myenv/bin/activate
```

On **Windows**:
```
myenv\Scripts\activate
```

### 5. Installing Dependencies

Install the required Python libraries:

```
pip3 install -r requirements.txt
```

**Note**: The provided `requirements.txt` is configured for CPU usage, which may result in slow performance. For better performance, we strongly recommend using a CUDA-enabled GPU. Follow these additional steps for GPU support:

- Install CUDA from [NVIDIA's CUDA Downloads Page](https://developer.nvidia.com/cuda-downloads).
- Install PyTorch with CUDA 12.1 support by following instructions specific to your device at [PyTorch's Get Started Page](https://pytorch.org/get-started/locally/).

### 6. Adding Your Virtual Environment to Jupyter

Add your virtual environment to your Jupyter environment:

```
python -m ipykernel install --user --name=myenv
```

Replace `myenv` with the name of your virtual environment.

### 7. Launching JupyterLab

Launch JupyterLab:

```
jupyter lab
```

### 8. Running the Notebook

In JupyterLab, navigate to `demo.ipynb`. Make sure to switch the notebook's kernel to your virtual environment to access the installed dependencies.

You're now ready to run the code in `demo.ipynb`.
