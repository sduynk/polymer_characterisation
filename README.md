# PolyChar

Official codebase for "Advancements in Polymer Characterisation Analysis: A Laser-Based System Integrating Computer Vision for Polymer Characterisation" 

PolyChar is a simple laser-based platform that combines computer vision and deep learning models to classify the solubility of different polymeric compounds across a range of solvents. Using the results obtained from the solubility screening method, Hansen Solubility Parameters (HSP) of the polymers using are determined using an optimisation algorithm. Additionally, a Convolutional Neural Network regression model is also used to estimate the size of polymeric nanoparticles between 20-470 nm.



## Method

![Overview](./assets/overview.png)

## Installation

To install the repository, follow these steps:

1. **Clone the repository**:
    ```sh
    git clone https://github.com/yourusername/PolyChar.git
    cd PolyChar
    ```

2. **Create a conda environment**:
    ```sh
    conda create --name polychar_env python=3.9
    conda activate polychar_env
    ```

3. **Install the required dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

4. **Verify the installation**:
    ```sh
    python -m unittest discover
    ```


## Solubility Results

Running solubility.ipynb on the solubility dataset should give the following results (ommitting @3 and @2 metrics).

| Model             |F1 Score@4     | Accuracy@4    | Precision@4   | Recall@4      |
|--------------     |-----------    |--------       |----------     |----------     |
| ResNet18          | 0.868±0.045   | 0.901±0.037   | 0.865±0.054   | 0.886±0.047   |
| EfficientNet B0   | 0.853±0.055   | 0.891±0.043   | 0.850±0.063   | 0.876±0.061   |
| ConvNext Tiny     | 0.865±0.045	| 0.904±0.035	| 0.869±0.051	| 0.878±0.050   |


## HSP Results

## Particle Size Results

## Project Structure

The project is organized as follows:

```
PolyChar/
├── Solubility/
│   ├── dataloaders.py
│   ├── train_classifier.py
│   ├── results/
│   │   └── Summarize_results.ipynb
│   ├── solubility.ipynb
│   ├── hparam.ipynb
│   ├── models.py
│   └── utils.py
├── ParticleSize/
│   ├── dataloaders.py
│   ├── train_regression.py
│   ├── ps_models.py
│   ├── polynomial_regression.ipynb
│   ├── particle_size.ipynb
│   ├── figures.ipynb
│   └── utils.py
├── HSP/
│   ├──
├── README.md
└── requirements.txt
```

### Other Files
- **README.md**: This file, providing an overview of the project.
- **requirements.txt**: Lists the dependencies required to run the project.

