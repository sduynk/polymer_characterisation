# PolyChar

Official codebase for "Computer Vision for Polymer Characterisation using Lasers" 

PolyChar is a simple laser-based platform that combines computer vision and deep learning models to classify the solubility of different polymeric compounds across a range of solvents. Using the results obtained from the solubility screening method, Hansen Solubility Parameters (HSP) of the polymers using are determined using an optimisation algorithm. Additionally, a Convolutional Neural Network regression model is also used to estimate the size of polymeric nanoparticles between 20-470 nm.



## Method

![illustration of three methods](method.png)

(Designed to be compatible with light-colored backgrounds.)


## Installation

To install the repository, follow these steps (Note that this will install the CPU version of torch only):

1. **Clone the repository**:
    ```sh
    git clone https://github.com/sduynk/Polymer_characterisation.git

    cd Polymer_characterisation
    ```

2. **Create a conda environment and ensure it is activated**:
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

If you want to install torch with CUDA for GPU acceleration, afterwards you can do the following.

5. **Uninstall torch and torchvision from the environment**
    ```
    pip uninstall torch torchvision
    ```

6. **Install torch and torchvision with CUDA support e.g. `cu121` (see notes for CUDA version):**

    ```sh
    pip install torch==2.3.0+cu121 torchvision==0.18.0+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
    ```


> **Note:**  
> - The CUDA version (e.g. `cu121`) should match your GPU and driver.  
> - For most users, you do **not** need to install the CUDA toolkit separately.
> - If you have a different GPU or CUDA version, see [PyTorch's official installation guide](https://pytorch.org/get-started/locally/).
> - While efforts have been made towards bit-accurate reproducibility, differences in drivers and cuda toolkits, among other factors, will likely yield slightly different results.

## Datasets

The necessary datasets to run the code are hosted at https://zenodo.org/records/15480040. Replace the empty folders for...
- `Particle Size Regression/Particle_Size-data`
- `Solubility Classification/Solubility-data`
- `Hansen Solubility Parameters/Hansen_Solubility_Parameters-data`

With the corresponding folders in Zenodo and you should be good to go!


## Solubility Results

Running `solubility.ipynb` on the solubility dataset should give the following **(or similar)** results (ommitting @3 and @2 metrics).

| Model             |F1 Score@4     | Accuracy@4    | Precision@4   | Recall@4      |
|--------------     |-----------    |--------       |----------     |----------     |
| ResNet18          | 0.868±0.045   | 0.901±0.037   | 0.865±0.054   | 0.886±0.047   |
| EfficientNet B0   | 0.853±0.055   | 0.891±0.043   | 0.850±0.063   | 0.876±0.061   |
| ConvNext Tiny     | 0.865±0.045	| 0.904±0.035	| 0.869±0.051	| 0.878±0.050   |


## Hansen Solubility Parameters Results

To obtain HSP Results, cd into the `Hansen Solubility Parameters` directory and run `python Genetic_algorithm.py`

| Polymer | Conc. (% w/v) | δD (GT) | δP (GT) | δH (GT) | δD (Pred) | δP (Pred) | δH (Pred) | R₀   | ED  | PED (%) |
| ------- | ------------- | ------- | ------- | ------- | --------- | --------- | --------- | ---- | --- | ------- |
| PMMA    | 5             | 18.6    | 10.5    | 5.1     | 17.4      | 10.4      | 3.1       | 9.2  | 2.4 | 11      |
| PS      | 5             | 18.5    | 4.5     | 2.9     | 18.1      | 3.9       | 5.7       | 4.5  | 2.9 | 15      |
| PVP     | 5             | 17.5    | 8.0     | 15.0    | 20.0      | 12.6      | 14.1      | 13.4 | 5.3 | 22      |
| PCL     | 5             | 17.7    | 5.0     | 8.4     | 18.3      | 10.5      | 5.0       | 9.6  | 6.5 | 32      |


## Particle Size Results

- Running `particle_size.ipynb` should give the following **(or similar)** results for **PPSNet - MLP (Sine)**.
- Running `polynomial_regression.ipynb` can be used to obtain the polynomial_regression results.

| Method                    | MAE (nm)(mean ± std)     | RMSE (nm)(mean ± std)     | R²(mean ± std)     |
| ------------------------- | ------------------------ | ------------------------- | ------------------ |
| **PPSNet - MLP (ReLU)**   | **9.53 ± 4.27**          | **15.60 ± 7.58**          | **0.99 ± 0.01**    |
| PPSNet (no conditioning)  | 22.25 ± 3.97             | 32.01 ± 6.95              | 0.93 ± 0.04        |
| Polynomial Regression     | 32.55 ± 6.67             | 47.81 ± 9.58              | 0.87 ± 0.03        |
| EfficientNet - MLP (Sine) | 11.60 ± 3.07             | 20.13 ± 5.95              | 0.98 ± 0.01        |



## Project Structure

The project is organized as follows:

```
PolyChar/
├── Solubility Classification/
│   ├── Solubility-Data/        # Data found in Zenodo
│   ├── results/                # ^
│   ├── Trained_Models/         # ^
│   ├── dataloaders.py 
│   ├── models.py 
│   ├── solubility.ipynb
│   ├── summarize_results.ipynb
│   ├── train.py 
│   └── utils.py 
├── ParticleSize Regression/
│   ├── Particle_Size-Data/     # Data found in Zenodo
│   ├── results/                # ^
│   ├── Trained_Models/         # ^
│   ├── dataloaders.py 
│   ├── figures.ipynb
│   ├── particle_size.ipynb
│   ├── polynomial_regression.ipynb
│   ├── ps_models.py
│   ├── train_regression.py 
│   └── utils.py 
├── Hansen Solubility Parameters/
│   ├── Genetic_algorithm.py
├── README.md
└── requirements.txt
```

### Other Files
- **README.md**: This file, providing an overview of the project.
- **requirements.txt**: Lists the dependencies required to run the project.


## Authors
George Killick (george.killick@liverpool.ac.uk)
Seda Uyanik (seda.uyanik@liverpool.ac.uk)


## License
Distributed under the Unlicense License. See LICENSE.txt for more information.
